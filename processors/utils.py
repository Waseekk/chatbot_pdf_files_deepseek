import streamlit as st

def _is_table(block):
    """Helper function to identify if a text block might be a table."""
    try:
        # PyMuPDF blocks are tuples with (x0, y0, x1, y1, "text", block_no, block_type)
        if isinstance(block, tuple) and len(block) >= 7:
            text = block[4]  # Get the text content
            if not isinstance(text, str):
                return False
                
            lines = text.split('\n')
            if len(lines) <= 2:
                return False
                
            # Check for regular structure in lines
            word_counts = [len(line.split()) for line in lines if line.strip()]
            if not word_counts:
                return False
                
            # Check if most lines have similar number of elements
            avg_count = sum(word_counts) / len(word_counts)
            deviation = sum(abs(c - avg_count) for c in word_counts) / len(word_counts)
            
            return deviation < 2 and avg_count > 2
            
    except (AttributeError, IndexError, TypeError):
        return False
        
    return False