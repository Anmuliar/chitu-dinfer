
import torch

class TokenArray:
    """ A token array to support read, update and expansion.

    We need to access the tokens that have been generated and write new tokens to the array.
    Some algorithms require to expand the token array.

    Parameters
    ----------
    prompt : Torch.Tensor
        The array that contains the input prompt. Can be:
        - 2D tensor of shape (batch_size, prompt_length) for standard usage
        - 1D tensor (packed format) when used with offset parameter
    gen_length : int
        The number of tokens to be generated.
    mask_id : int
        the mask id of the masked tokens
    eos_id : int
        the eos id of the end-of-sequence tokens
    device : Torch.Device
        The device where the token array is placed on.
    offset : List[int] or None
        When prompt is 1D packed format, offset specifies the length of each sequence.
        This is used to unpack the 1D payload into a 2D batch format.
    """
    def __init__(self, prompt, gen_length, mask_id, eos_id, device, offset=None):
        self.mask_id = mask_id
        self.eos_id = eos_id

        if offset is not None:
            # 1D packed payload mode
            # prompt is a 1D tensor containing multiple sequences packed together
            # offset is a list of lengths for each sequence
            self._offset = offset
            self._is_packed = True
            prompt = prompt.to(device)

            # Calculate batch size and max sequence length
            batch_size = len(offset)
            max_prompt_len = max(offset) if offset else 0

            # Create 2D data array with padding
            self.data = torch.full((batch_size, max_prompt_len + gen_length), mask_id, dtype=torch.long, device=device)

            # Unpack the 1D prompt into 2D format
            start_idx = 0
            for i, seq_len in enumerate(offset):
                if seq_len > 0:
                    self.data[i, :seq_len] = prompt[start_idx:start_idx + seq_len]
                    start_idx += seq_len

            # Store prompt as 2D for consistency with other methods
            self.prompt = self.data[:, :max_prompt_len].clone()
        else:
            # Standard 2D prompt mode
            self._offset = None
            self._is_packed = False
            self.prompt = prompt.to(device)
            self.data = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
            self.data[:, :prompt.shape[1]] = prompt.clone()

        self.gen_length = gen_length

    @property
    def total_length(self):
        return self.prompt.shape[1] + self.gen_length

    @property
    def batch_size(self):
        return self.prompt.shape[0]

    @property
    def device(self):
        return self.data.device

    @property
    def offset(self):
        return self._offset

    @property
    def is_packed(self):
        return self._is_packed

    def expand(self, new_len):
        pass

    def get_generated_tokens(self):
        if self.batch_size == 1:
            return self.data[self.data != self.eos_id].unsqueeze(0)
        else:
            self.data[self.data == self.mask_id] = self.eos_id
            return self.data

    def select_seqs(self, idx):
        arr = copy.copy(self)
        arr.prompt = self.prompt[idx]
        arr.data = self.data[idx]
        if self._offset is not None:
            if isinstance(idx, int):
                arr._offset = [self._offset[idx]]
            elif isinstance(idx, (list, tuple, slice)) or hasattr(idx, '__iter__'):
                arr._offset = [self._offset[i] for i in idx] if not isinstance(idx, slice) else self._offset[idx]
        return arr

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, vals):
        self.data[idx] = vals