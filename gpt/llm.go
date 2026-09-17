package gpt

import (
	"fmt"
	"strings"
)

// Supported LLM providers.
const (
	PROVIDEROPENAI    = "openai"
	PROVIDERANTHROPIC = "anthropic"
)

// LLM is the interface every chat provider implements. It is deliberately
// narrow: agents only ever need to ask a question and get text back.
type LLM interface {
	GptQuery(systemPrompt string, message string, context string) (string, error)
	SetApiKey(key string)
	SetModel(model string)
	SetURL(url string)
}

// Both providers must satisfy LLM.
var (
	_ LLM = (*OpenAI)(nil)
	_ LLM = (*Claude)(nil)
)

// DetectProvider infers the provider from a model name, so existing configs
// that only set "model" keep working without naming a provider.
func DetectProvider(model string) string {
	if strings.HasPrefix(strings.ToLower(model), "claude-") {
		return PROVIDERANTHROPIC
	}
	return PROVIDEROPENAI
}

// GetDefaultModelFor returns the default model for a provider.
func GetDefaultModelFor(provider string) string {
	switch strings.ToLower(provider) {
	case PROVIDERANTHROPIC:
		return MODELCLAUDE
	default:
		return MODELGPT35
	}
}

// NewProvider builds an LLM for the named provider. An empty provider is
// inferred from the model name, and an empty model falls back to that
// provider's default.
func NewProvider(provider string, key string, model string) (LLM, error) {
	if provider == "" {
		provider = DetectProvider(model)
	}
	provider = strings.ToLower(provider)

	if model == "" {
		model = GetDefaultModelFor(provider)
	}

	switch provider {
	case PROVIDEROPENAI:
		var o OpenAI
		o.SetApiKey(key)
		o.SetModel(model)
		return &o, nil
	case PROVIDERANTHROPIC:
		var c Claude
		c.SetApiKey(key)
		c.SetModel(model)
		return &c, nil
	default:
		return nil, fmt.Errorf("unknown llm provider %q, expected %q or %q", provider, PROVIDEROPENAI, PROVIDERANTHROPIC)
	}
}
