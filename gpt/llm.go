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

// Tunable knobs, named so a rejection can say which one the provider turned
// down.
//
// No knob works on every model: both providers removed the sampling params
// from their newest models and replaced them with an effort dial, so the two
// are very nearly mutually exclusive, and which side of the line a given
// model falls on changes with every release.
//
// Nothing here checks that up front. A table of which model accepts which
// parameter is stale the day a provider ships, and a stale table is worse
// than none: it refuses a config that would have worked. The provider is the
// authority, so knobs go out as configured and a rejection is explained
// after the fact; see explainKnobRejection.
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
	// ToolCalls holds the tools the model wants run. It is populated only
	// when StopReason is STOPTOOLUSE, and only on a request that offered
	// tools in the first place.
	ToolCalls []ToolCall
}

// Truncated reports whether the reply was cut short by the token ceiling,
// which is the signal to ask for a continuation.
func (r *Response) Truncated() bool {
	return r.StopReason == STOPMAXTOKENS
}

// WantsTool reports whether the model is waiting on a tool result. The reply
// text, if any, is a preamble to the call rather than an answer, and the
// conversation only continues once the results go back.
func (r *Response) WantsTool() bool {
	return r.StopReason == STOPTOOLUSE && len(r.ToolCalls) > 0
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
	//
	// It is not checked against the model. A model that no longer accepts
	// the parameter rejects the request, and that rejection is reported
	// with the knob named.
	SetTemperature(temperature *float64)
	// SetEffort sets reasoning depth, using one of the EFFORT* constants. An
	// empty string leaves it unset. Like SetTemperature it is not checked
	// against the model, only against the set of known levels.
	SetEffort(effort string)
	// NewChat starts a multi-turn conversation with a fixed system prompt
	// and tool set. It is the only entry point that can carry tools,
	// because answering a tool call needs a second request that still
	// remembers the first; GptQuery and Query are single-turn and cannot.
	// A nil or empty tool list is allowed and just makes it a plain chat.
	//
	// The knobs set on the provider apply to every turn of the chat.
	NewChat(systemPrompt string, tools []Tool) Chat
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

// ValidEffort reports whether a string is one of the EFFORT* levels.
//
// This is the one piece of validation kept up front, because the level set is
// a fixed vocabulary rather than a per-model capability: a typo like "hihg"
// is a config mistake at any model, and catching it at load costs nothing in
// staleness. Whether the chosen model accepts a valid level is the provider's
// call, not ours.
func ValidEffort(effort string) bool {
	switch effort {
	case EFFORTLOW, EFFORTMEDIUM, EFFORTHIGH, EFFORTXHIGH, EFFORTMAX:
		return true
	}
	return false
}

// ApplyKnobs sets the optional tuning knobs on an already-built provider.
//
// It does not ask whether the model accepts them. Which models take
// temperature and which take effort moves with every provider release, so
// the pairing is left to whoever wrote the config and the provider settles
// it on the first request; a rejection comes back through
// explainKnobRejection naming the knob.
//
// The one thing still refused here is an effort level that is not a level at
// all, which no release can turn into a valid value.
//
// A nil temperature and an empty effort are "not configured" and are skipped.
func ApplyKnobs(llm LLM, temperature *float64, effort string) error {
	if temperature != nil {
		llm.SetTemperature(temperature)
	}

	if effort != "" {
		if !ValidEffort(effort) {
			return fmt.Errorf("invalid %s %q, expected one of %s, %s, %s, %s, %s", KNOBEFFORT, effort, EFFORTLOW, EFFORTMEDIUM, EFFORTHIGH, EFFORTXHIGH, EFFORTMAX)
		}
		llm.SetEffort(effort)
	}

	return nil
}

// knobWireNames maps a knob to the spellings a provider might use for it in
// an error message. Effort has two because the providers put it in different
// places on the request: OpenAI as top-level reasoning_effort, Anthropic
// nested under output_config.
var knobWireNames = map[string][]string{
	KNOBTEMPERATURE: {"temperature"},
	KNOBEFFORT:      {"reasoning_effort", "effort"},
}

// explainKnobRejection annotates a provider error that looks like the refusal
// of a knob this request carried, naming the knob, the model and where the
// setting came from.
//
// This is the other half of not validating up front: the provider decides,
// but its own message ("Unsupported parameter: 'temperature'") says nothing
// about which config key to go and change. Wrapping it does.
//
// A knob only matches when it was actually configured AND the provider's text
// mentions it, so an expired key or a rate limit is returned untouched rather
// than blamed on a knob that had nothing to do with it.
func explainKnobRejection(err error, model string, temperature *float64, effort string) error {
	if err == nil {
		return nil
	}

	configured := map[string]bool{
		KNOBTEMPERATURE: temperature != nil,
		KNOBEFFORT:      effort != "",
	}

	text := strings.ToLower(err.Error())
	for _, knob := range []string{KNOBTEMPERATURE, KNOBEFFORT} {
		if !configured[knob] {
			continue
		}
		for _, name := range knobWireNames[knob] {
			if !strings.Contains(text, name) {
				continue
			}
			return fmt.Errorf("model %q rejected %q (set in the gpt config block): %w\n\nDrop the setting or pick a model that accepts it", model, knob, err)
		}
	}

	return err
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
