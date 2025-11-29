package markov

import (
	"math/rand"
	"regexp"
	"strings"
	"unicode"
)

// TextGenerator generates text using a Markov chain.
type TextGenerator struct {
	Chain        *Chain
	EndTokens    map[string]bool
	PreserveCase bool
}

// NewTextGenerator creates a new text generator with the specified n-gram size.
func NewTextGenerator(n int) *TextGenerator {
	return &TextGenerator{
		Chain: NewChain(n),
		EndTokens: map[string]bool{
			".": true,
			"!": true,
			"?": true,
		},
		PreserveCase: false,
	}
}

// tokenize splits text into tokens.
func (g *TextGenerator) tokenize(text string) []string {
	if !g.PreserveCase {
		text = strings.ToLower(text)
	}

	// Split on whitespace and handle punctuation
	words := strings.Fields(text)
	var tokens []string

	for _, word := range words {
		parts := splitPunctuation(word)
		for _, part := range parts {
			if part != "" {
				tokens = append(tokens, part)
			}
		}
	}

	return tokens
}

// splitPunctuation splits a word into parts with punctuation separated.
func splitPunctuation(word string) []string {
	var result []string
	var current strings.Builder

	runes := []rune(word)
	for i, r := range runes {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			current.WriteRune(r)
		} else {
			if current.Len() > 0 {
				result = append(result, current.String())
				current.Reset()
			}
			result = append(result, string(r))
		}

		// Handle end of word
		if i == len(runes)-1 && current.Len() > 0 {
			result = append(result, current.String())
		}
	}

	return result
}

// Train trains the generator on a text corpus.
func (g *TextGenerator) Train(text string) {
	tokens := g.tokenize(text)
	if len(tokens) > 0 {
		g.Chain.Train(tokens)
	}
}

// TrainMany trains on multiple texts.
func (g *TextGenerator) TrainMany(texts []string) {
	for _, text := range texts {
		g.Train(text)
	}
}

// Generate generates text with a maximum number of words.
func (g *TextGenerator) Generate(maxWords int, rng *rand.Rand) string {
	tokens := g.Chain.Generate(maxWords, rng)
	return g.joinTokens(tokens)
}

// GenerateFrom generates text starting with a specific prompt.
func (g *TextGenerator) GenerateFrom(prompt string, maxWords int, rng *rand.Rand) string {
	promptTokens := g.tokenize(prompt)

	if len(promptTokens) < g.Chain.Order {
		return g.Generate(maxWords, rng)
	}

	result := make([]string, len(promptTokens))
	copy(result, promptTokens)

	for len(result) < len(promptTokens)+maxWords {
		current := result[len(result)-g.Chain.Order:]
		next, ok := g.Chain.SampleNext(current, rng)
		if !ok {
			break
		}
		result = append(result, next)
		if g.EndTokens[next] {
			break
		}
	}

	return g.joinTokens(result)
}

// GenerateSentence generates a complete sentence.
func (g *TextGenerator) GenerateSentence(maxWords int, rng *rand.Rand) string {
	start, ok := g.Chain.SampleStart(rng)
	if !ok {
		return ""
	}

	result := make([]string, len(start))
	copy(result, start)

	for len(result) < maxWords {
		current := result[len(result)-g.Chain.Order:]
		next, ok := g.Chain.SampleNext(current, rng)
		if !ok {
			break
		}
		result = append(result, next)
		if g.EndTokens[next] {
			break
		}
	}

	return g.joinTokens(result)
}

// joinTokens joins tokens back into text with proper spacing.
func (g *TextGenerator) joinTokens(tokens []string) string {
	if len(tokens) == 0 {
		return ""
	}

	punctuation := regexp.MustCompile(`^[.!?,;:'")\]]$`)
	var result strings.Builder

	for i, token := range tokens {
		if i > 0 && !punctuation.MatchString(token) {
			result.WriteString(" ")
		}
		result.WriteString(token)
	}

	return result.String()
}

// Stats returns statistics about the generator.
func (g *TextGenerator) Stats() TextGeneratorStats {
	return TextGeneratorStats{
		Order:          g.Chain.Order,
		NumStates:      g.Chain.NumStates(),
		NumTransitions: g.Chain.NumTransitions(),
		NumSequences:   g.Chain.NumSequences,
		Entropy:        g.Chain.Entropy(),
	}
}

// TextGeneratorStats holds statistics about a text generator.
type TextGeneratorStats struct {
	Order          int
	NumStates      int
	NumTransitions int
	NumSequences   int
	Entropy        float64
}
