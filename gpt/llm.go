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

// Normalised stop reasons. Providers spell these differently, so they are
// mapped onto a common set that callers can branch on without caring which
// provider answered.
const (
	STOPEND       = "end"        // the model finished on its own
	STOPMAXTOKENS = "max_tokens" // the token ceiling was hit, the reply is cut short
	STOPTOOLUSE   = "tool_use"   // the model wants to call a tool
	STOPREFUSAL   = "refusal"    // the provider declined to answer
	STOPOTHER     = "other"      // anything the providers add later
)

// Response is a reply plus why generation stopped. Callers that need to
// continue a truncated reply, or react to a refusal, use this instead of the
// plain text returned by GptQuery.
type Response struct {
	Text string
	// StopReason is one of the STOP* constants above.
	StopReason string
	// RawStopReason is the provider's own value, kept for logging and for
	// reasons that have no normalised equivalent.
	RawStopReason string
	// Detail carries any provider explanation, such as why a request was
	// refused. It is usually empty.
	Detail string
}

// Truncated reports whether the reply was cut short by the token ceiling,
// which is the signal to ask for a continuation.
func (r *Response) Truncated() bool {
	return r.StopReason == STOPMAXTOKENS
}

// LLM is the interface every chat provider implements.
type LLM interface {
	// GptQuery returns just the reply text. A refusal or an empty reply is
	// reported as an error.
	GptQuery(systemPrompt string, message string, context string) (string, error)
	// Query returns the reply along with the stop reason, so callers can
	// detect truncation and build continuation loops. A refusal or an empty
	// reply comes back as a Response, not an error.
	Query(systemPrompt string, message string, context string) (*Response, error)
	SetApiKey(key string)
	SetModel(model string)
	SetURL(url string)
	// SetMaxTokens caps the reply length. Zero or less means the provider
	// default is used.
	SetMaxTokens(maxTokens int64)
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

// MaxTokens reports the configured reply ceiling. Zero means the provider
// default applies.
func (o *OpenAI) MaxTokens() int64 { return o.maxTokens }

// MaxTokens reports the configured reply ceiling. Zero means the provider
// default applies.
func (c *Claude) MaxTokens() int64 { return c.maxTokens }
