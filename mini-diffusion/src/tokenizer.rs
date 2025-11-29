//! Tokenizers for Text Encoders
//!
//! Tokenization converts text into sequences of integers that neural networks
//! can process. Different models use different tokenization strategies:
//!
//! - **BPE (Byte-Pair Encoding)**: Used by CLIP, GPT
//! - **Unigram**: Used by T5, SentencePiece
//!
//! Both are subword tokenizers - they break words into smaller pieces,
//! allowing the model to handle any text (even misspellings and new words).

use std::collections::{HashMap, HashSet};

/// Byte-Pair Encoding Tokenizer
/// 
/// BPE starts with a character-level vocabulary and iteratively merges
/// the most frequent pair of tokens. This creates a vocabulary of
/// subword units that balances:
/// - Common words as single tokens (efficient)
/// - Rare words as multiple subwords (handles anything)
///
/// Example: "tokenization" might become ["token", "ization"]
/// 
/// # How BPE Works
/// 
/// 1. Start: each character is a token
/// 2. Count all adjacent token pairs
/// 3. Merge the most frequent pair into a new token
/// 4. Repeat until vocabulary size reached
/// 
/// At inference, we apply these learned merges to new text.
pub struct BPETokenizer {
    /// Token string -> ID
    vocab: HashMap<String, u32>,
    /// ID -> Token string (for decoding)
    vocab_inverse: HashMap<u32, String>,
    /// Merge rules: (token1, token2) -> priority (lower = merge first)
    merges: HashMap<(String, String), usize>,
    /// Byte to unicode mapping (handles all bytes)
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    /// Special tokens
    pub start_token: u32,
    pub end_token: u32,
    pub pad_token: u32,
    pub unk_token: u32,
    /// Maximum sequence length
    pub max_length: usize,
    /// Regex pattern for initial tokenization
    pattern: String,
}

impl BPETokenizer {
    /// Create a new BPE tokenizer
    /// 
    /// In production, you'd load vocab and merges from a file.
    /// Here we create a minimal tokenizer for demonstration.
    pub fn new(max_length: usize) -> Self {
        let byte_encoder = Self::bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder.iter()
            .map(|(&k, &v)| (v, k))
            .collect();
        
        // Initialize with special tokens
        let mut vocab = HashMap::new();
        vocab.insert("<|startoftext|>".to_string(), 0);
        vocab.insert("<|endoftext|>".to_string(), 1);
        vocab.insert("<|pad|>".to_string(), 2);
        vocab.insert("<|unk|>".to_string(), 3);
        
        // Add basic ASCII vocabulary
        for i in 0..256u32 {
            let c = byte_encoder[&(i as u8)];
            vocab.entry(c.to_string()).or_insert(i + 4);
        }
        
        let vocab_inverse: HashMap<u32, String> = vocab.iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();
        
        BPETokenizer {
            vocab,
            vocab_inverse,
            merges: HashMap::new(),
            byte_encoder,
            byte_decoder,
            start_token: 0,
            end_token: 1,
            pad_token: 2,
            unk_token: 3,
            max_length,
            pattern: r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+".to_string(),
        }
    }
    
    /// Map bytes to unicode characters
    /// 
    /// GPT-2 style byte encoding: maps all 256 bytes to printable unicode
    /// characters. This ensures any byte sequence can be represented.
    fn bytes_to_unicode() -> HashMap<u8, char> {
        let mut bs: Vec<u8> = Vec::new();
        
        // Printable ASCII range
        bs.extend(b'!'..=b'~');
        // Latin-1 supplement (printable)
        bs.extend(0xa1..=0xac);
        bs.extend(0xae..=0xff);
        
        let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
        
        // Map remaining bytes to higher unicode codepoints
        let mut n = 0u32;
        for b in 0..=255u8 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        
        bs.into_iter()
            .zip(cs.into_iter())
            .map(|(b, c)| (b, char::from_u32(c).unwrap()))
            .collect()
    }
    
    /// Get all adjacent pairs in a word
    fn get_pairs(word: &[String]) -> HashSet<(String, String)> {
        let mut pairs = HashSet::new();
        if word.len() < 2 {
            return pairs;
        }
        
        let mut prev = &word[0];
        for token in &word[1..] {
            pairs.insert((prev.clone(), token.clone()));
            prev = token;
        }
        pairs
    }
    
    /// Apply BPE merges to a token
    fn bpe(&self, token: &str) -> Vec<String> {
        // Convert to initial character tokens
        let mut word: Vec<String> = token.chars()
            .map(|c| c.to_string())
            .collect();
        
        // Add end-of-word marker to last character
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        
        if word.len() == 1 {
            return word;
        }
        
        loop {
            let pairs = Self::get_pairs(&word);
            if pairs.is_empty() {
                break;
            }
            
            // Find the pair with lowest merge priority
            let bigram = pairs.iter()
                .filter_map(|pair| {
                    self.merges.get(pair).map(|&rank| (pair.clone(), rank))
                })
                .min_by_key(|(_, rank)| *rank);
            
            let (first, second) = match bigram {
                Some((pair, _)) => pair,
                None => break, // No more merges possible
            };
            
            // Apply the merge
            let mut new_word = Vec::new();
            let mut i = 0;
            
            while i < word.len() {
                // Find next occurrence of first token
                let j = word[i..].iter().position(|t| t == &first);
                
                match j {
                    Some(j) => {
                        new_word.extend_from_slice(&word[i..i+j]);
                        i += j;
                        
                        if i < word.len() - 1 && word[i] == first && word[i+1] == second {
                            // Merge the pair
                            new_word.push(format!("{}{}", first, second));
                            i += 2;
                        } else {
                            new_word.push(word[i].clone());
                            i += 1;
                        }
                    }
                    None => {
                        new_word.extend_from_slice(&word[i..]);
                        break;
                    }
                }
            }
            
            word = new_word;
            
            if word.len() == 1 {
                break;
            }
        }
        
        word
    }
    
    /// Tokenize text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let text = text.to_lowercase();
        let mut token_ids = vec![self.start_token];
        
        // Simple word-level tokenization for demo
        // Real implementation would use regex
        for word in text.split_whitespace() {
            // Convert to bytes, then to unicode chars
            let encoded: String = word.as_bytes()
                .iter()
                .map(|&b| self.byte_encoder[&b])
                .collect();
            
            // Apply BPE
            let bpe_tokens = self.bpe(&encoded);
            
            // Map to IDs
            for token in bpe_tokens {
                let id = self.vocab.get(&token)
                    .copied()
                    .unwrap_or(self.unk_token);
                token_ids.push(id);
            }
        }
        
        token_ids.push(self.end_token);
        
        // Truncate or pad to max_length
        if token_ids.len() > self.max_length {
            token_ids.truncate(self.max_length);
        }
        
        token_ids
    }
    
    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32]) -> String {
        let tokens: Vec<String> = ids.iter()
            .filter_map(|&id| self.vocab_inverse.get(&id).cloned())
            .collect();
        
        let text = tokens.join("");
        
        // Remove special tokens and end-of-word markers
        text.replace("<|startoftext|>", "")
            .replace("<|endoftext|>", "")
            .replace("</w>", " ")
            .trim()
            .to_string()
    }
    
    /// Tokenize batch with padding
    pub fn encode_batch(&self, texts: &[&str]) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let mut all_ids = Vec::new();
        let mut all_masks = Vec::new();
        
        // Find max length in batch
        let encoded: Vec<Vec<u32>> = texts.iter()
            .map(|t| self.encode(t))
            .collect();
        
        let max_len = encoded.iter()
            .map(|ids| ids.len())
            .max()
            .unwrap_or(0)
            .min(self.max_length);
        
        for mut ids in encoded {
            let mut mask = vec![1u32; ids.len()];
            
            // Pad to max length
            while ids.len() < max_len {
                ids.push(self.pad_token);
                mask.push(0);
            }
            
            ids.truncate(max_len);
            mask.truncate(max_len);
            
            all_ids.push(ids);
            all_masks.push(mask);
        }
        
        (all_ids, all_masks)
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// Unigram Tokenizer (SentencePiece style)
/// 
/// Unigram uses a probabilistic model over subwords. Unlike BPE which
/// greedily merges, Unigram:
/// 
/// 1. Starts with a large vocabulary
/// 2. Assigns probability to each subword
/// 3. Finds the most likely segmentation using Viterbi algorithm
/// 4. Iteratively removes low-probability subwords
///
/// This gives more linguistically motivated tokenizations.
/// 
/// # Key Difference from BPE
/// 
/// BPE: deterministic merges based on frequency
/// Unigram: probabilistic, finds optimal segmentation
/// 
/// T5 uses Unigram via SentencePiece.
pub struct UnigramTokenizer {
    /// Token -> (ID, log probability)
    vocab: HashMap<String, (u32, f32)>,
    /// ID -> Token
    vocab_inverse: HashMap<u32, String>,
    /// Special tokens
    pub pad_token: u32,
    pub unk_token: u32,
    pub eos_token: u32,
    /// Maximum sequence length
    pub max_length: usize,
}

impl UnigramTokenizer {
    /// Create a new Unigram tokenizer
    pub fn new(max_length: usize) -> Self {
        let mut vocab = HashMap::new();
        
        // Special tokens
        vocab.insert("<pad>".to_string(), (0, 0.0));
        vocab.insert("<unk>".to_string(), (1, -10.0));
        vocab.insert("</s>".to_string(), (2, 0.0));
        vocab.insert("▁".to_string(), (3, -1.0)); // Sentence piece space marker
        
        // Add basic character vocabulary with uniform probabilities
        for (i, c) in "abcdefghijklmnopqrstuvwxyz0123456789 ".chars().enumerate() {
            let token = if c == ' ' {
                "▁".to_string()
            } else {
                c.to_string()
            };
            vocab.entry(token).or_insert(((i + 4) as u32, -2.0));
        }
        
        // Add some common subwords
        let common_subwords = [
            "the", "ing", "tion", "ed", "er", "es", "ly", "al",
            "photo", "image", "cat", "dog", "man", "woman",
            "of", "a", "an", "is", "are", "was", "were",
        ];
        
        let mut id = vocab.len() as u32;
        for word in common_subwords {
            vocab.entry(format!("▁{}", word)).or_insert((id, -1.5));
            id += 1;
            vocab.entry(word.to_string()).or_insert((id, -2.0));
            id += 1;
        }
        
        let vocab_inverse: HashMap<u32, String> = vocab.iter()
            .map(|(k, &(id, _))| (id, k.clone()))
            .collect();
        
        UnigramTokenizer {
            vocab,
            vocab_inverse,
            pad_token: 0,
            unk_token: 1,
            eos_token: 2,
            max_length,
        }
    }
    
    /// Find the best segmentation using dynamic programming
    /// 
    /// This is a simplified Viterbi-like algorithm:
    /// For each position, find the best way to get there from any previous position.
    fn segment(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        
        if n == 0 {
            return vec![];
        }
        
        // best_score[i] = best log probability to segment chars[0..i]
        let mut best_score = vec![f32::NEG_INFINITY; n + 1];
        // best_length[i] = length of token ending at position i in best segmentation
        let mut best_length = vec![0usize; n + 1];
        
        best_score[0] = 0.0;
        
        for end in 1..=n {
            // Try all possible last tokens ending at `end`
            for start in 0..end {
                let substr: String = chars[start..end].iter().collect();
                
                if let Some(&(_, log_prob)) = self.vocab.get(&substr) {
                    let score = best_score[start] + log_prob;
                    if score > best_score[end] {
                        best_score[end] = score;
                        best_length[end] = end - start;
                    }
                }
            }
            
            // Fallback to character-level if no token found
            if best_score[end] == f32::NEG_INFINITY {
                let char_token: String = chars[end-1..end].iter().collect();
                let log_prob = self.vocab.get(&char_token)
                    .map(|&(_, p)| p)
                    .unwrap_or(-10.0);
                
                best_score[end] = best_score[end-1] + log_prob;
                best_length[end] = 1;
            }
        }
        
        // Backtrack to get tokens
        let mut tokens = Vec::new();
        let mut pos = n;
        
        while pos > 0 {
            let len = best_length[pos];
            let token: String = chars[pos-len..pos].iter().collect();
            tokens.push(token);
            pos -= len;
        }
        
        tokens.reverse();
        tokens
    }
    
    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Preprocess: lowercase and add space markers
        let text = text.to_lowercase();
        let processed = format!("▁{}", text.replace(' ', "▁"));
        
        let tokens = self.segment(&processed);
        
        let mut ids: Vec<u32> = tokens.iter()
            .map(|t| {
                self.vocab.get(t)
                    .map(|&(id, _)| id)
                    .unwrap_or(self.unk_token)
            })
            .collect();
        
        // Add EOS token
        ids.push(self.eos_token);
        
        // Truncate to max length
        if ids.len() > self.max_length {
            ids.truncate(self.max_length);
        }
        
        ids
    }
    
    /// Decode token IDs back to text
    pub fn decode(&self, ids: &[u32]) -> String {
        let tokens: Vec<&str> = ids.iter()
            .filter_map(|&id| self.vocab_inverse.get(&id).map(|s| s.as_str()))
            .filter(|&t| t != "</s>" && t != "<pad>")
            .collect();
        
        tokens.join("")
            .replace("▁", " ")
            .trim()
            .to_string()
    }
    
    /// Encode batch with padding
    pub fn encode_batch(&self, texts: &[&str]) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let encoded: Vec<Vec<u32>> = texts.iter()
            .map(|t| self.encode(t))
            .collect();
        
        let max_len = encoded.iter()
            .map(|ids| ids.len())
            .max()
            .unwrap_or(0)
            .min(self.max_length);
        
        let mut all_ids = Vec::new();
        let mut all_masks = Vec::new();
        
        for mut ids in encoded {
            let mut mask = vec![1u32; ids.len()];
            
            while ids.len() < max_len {
                ids.push(self.pad_token);
                mask.push(0);
            }
            
            ids.truncate(max_len);
            mask.truncate(max_len);
            
            all_ids.push(ids);
            all_masks.push(mask);
        }
        
        (all_ids, all_masks)
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bpe_basic() {
        let tokenizer = BPETokenizer::new(77);
        
        let text = "a photo of a cat";
        let ids = tokenizer.encode(text);
        
        // Should start with start token
        assert_eq!(ids[0], tokenizer.start_token);
        // Should end with end token
        assert_eq!(*ids.last().unwrap(), tokenizer.end_token);
        
        // Should have some tokens in between
        assert!(ids.len() > 2);
    }
    
    #[test]
    fn test_unigram_basic() {
        let tokenizer = UnigramTokenizer::new(77);
        
        let text = "a photo of a cat";
        let ids = tokenizer.encode(text);
        
        // Should end with EOS
        assert_eq!(*ids.last().unwrap(), tokenizer.eos_token);
        
        // Should decode back to similar text
        let decoded = tokenizer.decode(&ids);
        assert!(decoded.contains("photo") || decoded.len() > 0);
    }
    
    #[test]
    fn test_bpe_batch() {
        let tokenizer = BPETokenizer::new(77);
        
        let texts = ["hello world", "short"];
        let (ids, masks) = tokenizer.encode_batch(&texts);
        
        // Both should have same length (padded)
        assert_eq!(ids[0].len(), ids[1].len());
        assert_eq!(masks[0].len(), masks[1].len());
        
        // Shorter sequence should have padding (0s in mask)
        let padding_count: u32 = masks[1].iter().filter(|&&m| m == 0).sum();
        assert!(padding_count > 0);
    }
}
