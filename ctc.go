package tfhelper

import (
	"slices"
	"strings"
)

// CTCDecode decodes the models output to a string
func CTCDecode(output [][][]float32, alphabet string) []string {
	var texts []string

	for _, line := range output {
		var text strings.Builder
		var lastIdx int = -1

		for _, char := range line {
			maxIdx := slices.Index(char, slices.Max(char))
			if maxIdx != lastIdx && maxIdx < len(alphabet) {
				text.WriteRune(rune(alphabet[maxIdx]))
				lastIdx = maxIdx
			}
		}

		texts = append(texts, text.String())
	}

	return texts
}
