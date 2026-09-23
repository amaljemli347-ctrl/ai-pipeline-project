def recursive_character_split(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Recursively splits a text into chunks based on a list of separators.
    It tries to split by paragraph, then sentence, then words to keep chunks meaningful.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(text: str, separators: list[str]) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
            
        separator = separators[0]
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break
                
        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_len = len(split) + (len(separator) if current_length > 0 else 0)
            
            if current_length + split_len > chunk_size and current_length > 0:
                merged_chunk = separator.join(current_chunk)
                chunks.append(merged_chunk)
                
                # To calculate overlap, we just keep elements from the end of current_chunk
                # until their total length is <= chunk_overlap
                overlap_chunk = []
                overlap_length = 0
                for item in reversed(current_chunk):
                    item_len = len(item) + (len(separator) if overlap_length > 0 else 0)
                    if overlap_length + item_len > chunk_overlap:
                        break
                    overlap_chunk.insert(0, item)
                    overlap_length += item_len
                    
                current_chunk = overlap_chunk
                current_length = overlap_length
            
            current_chunk.append(split)
            current_length += len(split) + (len(separator) if current_length > 0 else 0)
            
        if current_chunk:
            chunks.append(separator.join(current_chunk))
            
        final_chunks = []
        next_separators = separators[separators.index(separator) + 1:] if separator in separators else separators
        
        for chunk in chunks:
            if len(chunk) > chunk_size and next_separators:
                final_chunks.extend(split_text(chunk, next_separators))
            elif len(chunk) > chunk_size:
                # Forcefully truncate if no separators left
                for i in range(0, len(chunk), chunk_size - chunk_overlap):
                    final_chunks.append(chunk[i:i + chunk_size])
            else:
                final_chunks.append(chunk)
                
        return final_chunks

    if not text.strip():
        return []
        
    return split_text(text.strip(), separators)
