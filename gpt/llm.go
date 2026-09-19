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

// Tunable knobs, named so callers can ask Supports about them.
//
// No knob works on every model. Both providers removed the sampling params
// from their newest models and replaced them with an effort dial, so the two
// knobs are very nearly mutually exclusive:
//
//	                      temperature   effort
//	gpt-3.5 / gpt-4o          yes         no
//	o-series / gpt-5          no          yes
//	claude-3.x / haiku-4-5    yes         no
//	claude-opus-5             no          yes
//
// Setting one on a model that rejects it is a 400 from the provider, which is
// why NewLLM checks Supports at startup instead of finding out on the first
// Slack mention.
const (
	KNOBTEMPERATURE = "temperature"
	KNOBEFFORT      = "effort"
)

// Effort levels. OpenAI stops at high; the top two are Anthropic-only, and
// Anthropic's own older effort-capable models (Opus 4.5) only accept the
// first three. Providers reject a level they do not know.
const (
	EFFORTLOW    = "low"
	EFFORTMEDIUM = "medium"
	EFFORTHIGH   = "high"
	EFFORTXHIGH  = "xhigh"
	EFFORTMAX    = "max"
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
	// SetTemperature sets the sampling temperature. A nil value leaves it
	// unset, which is deliberately not the same as zero: zero is a valid
	// request for determinism, so the "0 means unset" shortcut used by
	// SetMaxTokens would make the two indistinguishable.
	//
	// The value is passed through provider-native and is NOT rescaled, so
	// the meaningful range differs: 0-2 on OpenAI (1 is its default), 0-1
	// on Anthropic. The same number is a different request to each.
	SetTemperature(temperature *float64)
	// SetEffort sets reasoning depth on models that support it, using one of
	// the EFFORT* constants. An empty string leaves it unset.
	SetEffort(effort string)
	// Supports reports whether the currently configured model accepts a
	// knob, so callers can fail at startup rather than on first use.
	Supports(knob string) bool
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

// ValidEffort reports whether a string is one of the EFFORT* levels. It does
// not check that the configured model accepts that particular level, only
// that the value is a level at all; Supports covers the model side.
func ValidEffort(effort string) bool {
	switch effort {
	case EFFORTLOW, EFFORTMEDIUM, EFFORTHIGH, EFFORTXHIGH, EFFORTMAX:
		return true
	}
	return false
}

// ApplyKnobs sets the optional tuning knobs on an already-built provider,
// refusing any the configured model does not accept. It exists so a bad
// combination is caught once at startup with a message naming the model,
// rather than as a 400 on the first query.
//
// A nil temperature and an empty effort are "not configured" and are skipped.
func ApplyKnobs(llm LLM, model string, temperature *float64, effort string) error {
	if temperature != nil {
		if !llm.Supports(KNOBTEMPERATURE) {
			return fmt.Errorf("model %q does not accept %s; it was removed on this model in favour of %s, so drop the temperature setting or pick an older model", model, KNOBTEMPERATURE, KNOBEFFORT)
		}
		llm.SetTemperature(temperature)
	}

	if effort != "" {
		if !ValidEffort(effort) {
			return fmt.Errorf("invalid %s %q, expected one of %s, %s, %s, %s, %s", KNOBEFFORT, effort, EFFORTLOW, EFFORTMEDIUM, EFFORTHIGH, EFFORTXHIGH, EFFORTMAX)
		}
		if !llm.Supports(KNOBEFFORT) {
			return fmt.Errorf("model %q does not accept %s; only reasoning-capable models do, so drop the effort setting or pick a newer model", model, KNOBEFFORT)
		}
		llm.SetEffort(effort)
	}

	return nil
}

// hasAnyPrefix reports whether model begins with any of the prefixes, ignoring
// case. The capability tables in claude.go and gpt.go are prefix lists because
// model names are versioned by suffix.
func hasAnyPrefix(model string, prefixes []string) bool {
	m := strings.ToLower(model)
	for _, p := range prefixes {
		if strings.HasPrefix(m, p) {
			return true
		}
	}
	return false
}

// MaxTokens reports the configured reply ceiling. Zero means the provider
// default applies.
func (o *OpenAI) MaxTokens() int64 { return o.maxTokens }

// MaxTokens reports the configured reply ceiling. Zero means the provider
// default applies.
func (c *Claude) MaxTokens() int64 { return c.maxTokens }

// Temperature reports the configured sampling temperature. Nil means unset,
// which is distinct from a configured zero.
func (o *OpenAI) Temperature() *float64 { return o.temperature }

// Temperature reports the configured sampling temperature. Nil means unset,
// which is distinct from a configured zero.
func (c *Claude) Temperature() *float64 { return c.temperature }

// Effort reports the configured reasoning depth. Empty means unset.
func (o *OpenAI) Effort() string { return o.effort }

// Effort reports the configured reasoning depth. Empty means unset.
func (c *Claude) Effort() string { return c.effort }
