package gpt

import (
	"errors"
	"strings"
	"testing"
)

func TestDetectProvider(t *testing.T) {
	cases := []struct {
		model string
		want  string
	}{
		{"claude-opus-5", PROVIDERANTHROPIC},
		{"claude-haiku-4-5", PROVIDERANTHROPIC},
		{"Claude-Opus-5", PROVIDERANTHROPIC},
		{"gpt-3.5-turbo", PROVIDEROPENAI},
		{"gpt-4o", PROVIDEROPENAI},
		{"", PROVIDEROPENAI},
	}
	for _, c := range cases {
		if got := DetectProvider(c.model); got != c.want {
			t.Errorf("DetectProvider(%q) = %q, want %q", c.model, got, c.want)
		}
	}
}

func TestNewProviderReturnsCorrectType(t *testing.T) {
	llm, err := NewProvider(PROVIDERANTHROPIC, "k", "claude-opus-5")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := llm.(*Claude); !ok {
		t.Errorf("expected *Claude, got %T", llm)
	}

	llm, err = NewProvider(PROVIDEROPENAI, "k", "gpt-3.5-turbo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := llm.(*OpenAI); !ok {
		t.Errorf("expected *OpenAI, got %T", llm)
	}
}

// An empty provider must be inferred from the model, so configs written before
// the provider field existed keep working.
func TestNewProviderInfersFromModel(t *testing.T) {
	llm, err := NewProvider("", "k", "claude-opus-5")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := llm.(*Claude); !ok {
		t.Errorf("empty provider with a claude model should give *Claude, got %T", llm)
	}

	llm, err = NewProvider("", "k", "gpt-3.5-turbo")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := llm.(*OpenAI); !ok {
		t.Errorf("empty provider with a gpt model should give *OpenAI, got %T", llm)
	}
}

func TestNewProviderDefaultsModel(t *testing.T) {
	llm, err := NewProvider(PROVIDERANTHROPIC, "k", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := llm.(*Claude).model; got != MODELCLAUDE {
		t.Errorf("model = %q, want %q", got, MODELCLAUDE)
	}

	llm, err = NewProvider(PROVIDEROPENAI, "k", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := llm.(*OpenAI).model; got != MODELGPT35 {
		t.Errorf("model = %q, want %q", got, MODELGPT35)
	}
}

func TestNewProviderRejectsUnknown(t *testing.T) {
	if _, err := NewProvider("gemini", "k", "whatever"); err == nil {
		t.Fatal("expected an error for an unknown provider, got nil")
	}
}

func TestProviderIsCaseInsensitive(t *testing.T) {
	if _, err := NewProvider("Anthropic", "k", ""); err != nil {
		t.Errorf("provider names should be case-insensitive: %v", err)
	}
}

func TestValidEffort(t *testing.T) {
	for _, e := range []string{EFFORTLOW, EFFORTMEDIUM, EFFORTHIGH, EFFORTXHIGH, EFFORTMAX} {
		if !ValidEffort(e) {
			t.Errorf("ValidEffort(%q) = false, want true", e)
		}
	}
	for _, e := range []string{"", "LOW", "highest", "0.7"} {
		if ValidEffort(e) {
			t.Errorf("ValidEffort(%q) = true, want false", e)
		}
	}
}

// A misspelled level is the one knob mistake still caught at load, because no
// provider release can turn "turbo" into a valid effort.
func TestApplyKnobsRejectsInvalidEffort(t *testing.T) {
	llm, _ := NewProvider(PROVIDERANTHROPIC, "k", "claude-opus-5")
	if err := ApplyKnobs(llm, nil, "turbo"); err == nil {
		t.Error("expected an error for an effort level that is not one of the EFFORT* constants")
	}
}

// ApplyKnobs deliberately does NOT know which model takes which parameter.
// Both of these pairings are ones the provider will reject, and both must be
// accepted here: the alternative is a capability table that refuses a config
// the day a provider changes its mind.
func TestApplyKnobsDoesNotJudgeTheModel(t *testing.T) {
	temp := 0.7

	claude, _ := NewProvider(PROVIDERANTHROPIC, "k", "claude-opus-5")
	if err := ApplyKnobs(claude, &temp, ""); err != nil {
		t.Fatalf("temperature must be passed through unjudged, got %v", err)
	}
	if got := claude.(*Claude).temperature; got == nil || *got != temp {
		t.Errorf("temperature = %v, want %v", got, temp)
	}

	openai, _ := NewProvider(PROVIDEROPENAI, "k", "gpt-3.5-turbo")
	if err := ApplyKnobs(openai, nil, EFFORTHIGH); err != nil {
		t.Fatalf("effort must be passed through unjudged, got %v", err)
	}
	if got := openai.(*OpenAI).effort; got != EFFORTHIGH {
		t.Errorf("effort = %q, want %q", got, EFFORTHIGH)
	}
}

func TestApplyKnobsSetsBothKnobs(t *testing.T) {
	temp := 0.3
	llm, _ := NewProvider(PROVIDEROPENAI, "k", "gpt-4o")
	if err := ApplyKnobs(llm, &temp, EFFORTMEDIUM); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	o := llm.(*OpenAI)
	if o.temperature == nil || *o.temperature != temp {
		t.Errorf("temperature = %v, want %v", o.temperature, temp)
	}
	if o.effort != EFFORTMEDIUM {
		t.Errorf("effort = %q, want %q", o.effort, EFFORTMEDIUM)
	}
}

// An unset knob must stay unset rather than being defaulted to a zero value,
// which is why the config field is a pointer.
func TestApplyKnobsLeavesUnsetKnobsAlone(t *testing.T) {
	llm, _ := NewProvider(PROVIDEROPENAI, "k", "gpt-4o")
	if err := ApplyKnobs(llm, nil, ""); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if o := llm.(*OpenAI); o.temperature != nil || o.effort != "" {
		t.Errorf("unset knobs should stay unset, got temperature=%v effort=%q", o.temperature, o.effort)
	}
}

// Zero is a real temperature, not an absent one. If this regresses, a config
// asking for deterministic output would silently send no temperature at all.
func TestApplyKnobsTreatsZeroTemperatureAsSet(t *testing.T) {
	zero := 0.0
	llm, _ := NewProvider(PROVIDEROPENAI, "k", "gpt-4o")
	if err := ApplyKnobs(llm, &zero, ""); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	got := llm.(*OpenAI).temperature
	if got == nil {
		t.Fatal("temperature 0 was dropped; it must be distinguishable from unset")
	}
	if *got != 0 {
		t.Errorf("temperature = %v, want 0", *got)
	}
}

// Nothing is validated before sending, so the rejection is the only warning
// anyone gets. It has to name the knob and say where it was set, because the
// provider's own text does neither.
func TestExplainKnobRejectionNamesTheKnob(t *testing.T) {
	temp := 0.7
	raw := errors.New("OpenAI API error: Unsupported parameter: 'temperature' is not supported with this model.")

	err := explainKnobRejection(raw, "gpt-5", &temp, "")
	if err == nil {
		t.Fatal("expected the error to be returned")
	}

	got := err.Error()
	for _, want := range []string{`"gpt-5"`, `"temperature"`, "gpt config block", "Unsupported parameter"} {
		if !strings.Contains(got, want) {
			t.Errorf("explanation is missing %q:\n%s", want, got)
		}
	}
	if !errors.Is(err, raw) {
		t.Error("the provider's own error must stay wrapped, not be replaced")
	}
}

// Anthropic nests effort under output_config and OpenAI sends it top-level as
// reasoning_effort, so both spellings have to be recognised.
func TestExplainKnobRejectionMatchesEitherEffortSpelling(t *testing.T) {
	for _, text := range []string{
		"OpenAI API error: Unrecognized request argument supplied: reasoning_effort",
		"anthropic: output_config.effort: Input should be 'low', 'medium' or 'high'",
	} {
		err := explainKnobRejection(errors.New(text), "some-model", nil, EFFORTMAX)
		if !strings.Contains(err.Error(), `rejected "effort"`) {
			t.Errorf("effort rejection not recognised in %q:\n%s", text, err)
		}
	}
}

// Blaming a knob for an unrelated failure would send someone editing the
// wrong config key, so a match needs both the knob set AND the provider
// naming it.
func TestExplainKnobRejectionLeavesUnrelatedErrorsAlone(t *testing.T) {
	temp := 0.7

	cases := []struct {
		name        string
		text        string
		temperature *float64
		effort      string
	}{
		{"unrelated failure with a knob set", "OpenAI API error: Incorrect API key provided", &temp, ""},
		{"knob named but not configured", "Unsupported parameter: 'temperature'", nil, ""},
		{"other knob named", "Unsupported parameter: 'temperature'", nil, EFFORTHIGH},
	}

	for _, c := range cases {
		raw := errors.New(c.text)
		got := explainKnobRejection(raw, "gpt-4o", c.temperature, c.effort)
		if got.Error() != c.text {
			t.Errorf("%s: error should pass through unchanged, got %q", c.name, got)
		}
	}
}

func TestExplainKnobRejectionIsCaseInsensitive(t *testing.T) {
	temp := 0.7
	err := explainKnobRejection(errors.New("Request rejected: TEMPERATURE is not permitted"), "m", &temp, "")
	if !strings.Contains(err.Error(), `rejected "temperature"`) {
		t.Errorf("provider casing should not matter:\n%s", err)
	}
}

func TestExplainKnobRejectionPassesNilThrough(t *testing.T) {
	temp := 0.7
	if err := explainKnobRejection(nil, "m", &temp, EFFORTHIGH); err != nil {
		t.Errorf("a successful request must stay successful, got %v", err)
	}
}

// The wrapper reports the model that was actually sent, which for an unset
// model is the provider default rather than an empty string.
func TestModelOrDefault(t *testing.T) {
	var c Claude
	if got := c.modelOrDefault(); got != MODELCLAUDE {
		t.Errorf("Claude.modelOrDefault() = %q, want %q", got, MODELCLAUDE)
	}
	c.SetModel("claude-haiku-4-5")
	if got := c.modelOrDefault(); got != "claude-haiku-4-5" {
		t.Errorf("Claude.modelOrDefault() = %q, want the configured model", got)
	}

	var o OpenAI
	if got := o.modelOrDefault(); got != MODELGPT35 {
		t.Errorf("OpenAI.modelOrDefault() = %q, want %q", got, MODELGPT35)
	}
}
