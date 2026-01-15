class RepetitionEncoder:
    """
    
    
    """
    
    def __init__(self, hidden_bits):
        """
        
        
        Args:
            hidden_bits: ，01， [0, 1, 1, 0, 1]
        """
        self.hidden_bits = hidden_bits
        self.index = 0
        self.length = len(hidden_bits)
        
    def get_current_bit(self):
        """
        bit
        index，
        
        Returns:
            int: bit (0  1)
        """
        if self.length == 0:
            raise ValueError("")
            
        current_bit = self.hidden_bits[self.index]
        self.index = (self.index + 1) % self.length  # 
        return current_bit
    
    def encode(self, repetition_count):
        """
        
        
        Args:
            repetition_count: bit
            
        Returns:
            list: 
        """
        encoded = []
        for bit in self.hidden_bits:
            encoded.extend([bit] * repetition_count)
        return encoded
    
    def reset_index(self):
        """"""
        self.index = 0
        
    def get_hidden_info(self):
        """"""
        return self.hidden_bits.copy()
    
    def get_current_index(self):
        """"""
        return self.index


class MajorityVotingDecoder:
    """
    
    
    """
    
    def __init__(self, hidden_length):
        """
        
        
        Args:
            hidden_length: （bit）
        """
        self.hidden_length = hidden_length
        
    def decode(self, encoded_bits):
        """
        
        
        Args:
            encoded_bits: 01
            
        Returns:
            list: 
        """
        if not encoded_bits:
            raise ValueError("")
            
        if len(encoded_bits) % self.hidden_length != 0:
            raise ValueError(f"({len(encoded_bits)})({self.hidden_length})")
            
        repetition_count = len(encoded_bits) // self.hidden_length
        decoded_bits = []
        
        # 
        for i in range(self.hidden_length):
            # ibit
            votes = []
            for j in range(repetition_count):
                votes.append(encoded_bits[j * self.hidden_length + i])
            # print(votes)
            # 
            decoded_bit = self._majority_vote(votes)
            decoded_bits.append(decoded_bit)
            
        return decoded_bits
    
    def _majority_vote(self, votes):
        """
        
        
        Args:
            votes: 
            
        Returns:
            int:  (0  1)
        """
        count_0 = votes.count(0)
        count_1 = votes.count(1)
        
        if count_1 > count_0:
            return 1
        elif count_0 > count_1:
            return 0
        else:
            # ，0（）
            return 0
    
    def decode_with_noise(self, encoded_bits, noise_positions=None):
        """
        （）
        
        Args:
            encoded_bits: 
            noise_positions: ，bit
            
        Returns:
            list: 
        """
        if noise_positions is None:
            return self.decode(encoded_bits)
            
        # 
        noisy_bits = encoded_bits.copy()
        
        # （bit）
        for pos in noise_positions:
            if 0 <= pos < len(noisy_bits):
                noisy_bits[pos] = 1 - noisy_bits[pos]
        
        return self.decode(noisy_bits)


# 
def demo():
    """"""
    print("===  ===")
    
    # 
    original_info = [1, 0, 1, 1, 0]
    print(f": {original_info}")
    
    # 
    encoder = RepetitionEncoder(original_info)
    
    # （bit3）
    repetition_count = 3
    encoded_sequence = encoder.encode(repetition_count)
    print(f"（{repetition_count}）: {encoded_sequence}")
    
    # 
    decoder = MajorityVotingDecoder(len(original_info))
    
    # 
    decoded_info = decoder.decode(encoded_sequence)
    print(f": {decoded_info}")
    print(f": {decoded_info == original_info}")
    
    print("\n===  ===")
    # 
    noise_positions = [2, 7, 11]  # 
    print(f": {noise_positions}")
    
    noisy_decoded = decoder.decode_with_noise(encoded_sequence, noise_positions)
    print(f": {noisy_decoded}")
    print(f": {noisy_decoded == original_info}")
    
    print("\n===  ===")
    # 
    encoder.reset_index()
    print("bit:")
    for i in range(12):  # 12bit，
        bit = encoder.get_current_bit()
        print(f"{i}: {bit}")


if __name__ == "__main__":
    demo()