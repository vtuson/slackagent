package gpt

import "testing"

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

func TestSupportsKnobsByModel(t *testing.T) {
	cases := []struct {
		provider    string
		model       string
		temperature bool
		effort      bool
	}{
		// Anthropic: sampling params were dropped on Opus 4.7 and after,
		// effort arrived on Opus 4.5.
		{PROVIDERANTHROPIC, "claude-opus-5", false, true},
		{PROVIDERANTHROPIC, "claude-opus-4-8", false, true},
		{PROVIDERANTHROPIC, "claude-opus-4-7", false, true},
		{PROVIDERANTHROPIC, "claude-sonnet-5", false, true},
		{PROVIDERANTHROPIC, "claude-fable-5-1", false, true},
		{PROVIDERANTHROPIC, "claude-opus-4-6", true, true},
		{PROVIDERANTHROPIC, "claude-sonnet-4-6", true, true},
		{PROVIDERANTHROPIC, "claude-haiku-4-5", true, false},
		{PROVIDERANTHROPIC, "claude-3-5-sonnet", true, false},
		// OpenAI: the reasoning families are the mirror image of the chat
		// families.
		{PROVIDEROPENAI, "gpt-3.5-turbo", true, false},
		{PROVIDEROPENAI, "gpt-4o", true, false},
		{PROVIDEROPENAI, "o3-mini", false, true},
		{PROVIDEROPENAI, "gpt-5", false, true},
	}

	for _, c := range cases {
		llm, err := NewProvider(c.provider, "k", c.model)
		if err != nil {
			t.Fatalf("NewProvider(%q): %v", c.model, err)
		}
		if got := llm.Supports(KNOBTEMPERATURE); got != c.temperature {
			t.Errorf("%s Supports(temperature) = %v, want %v", c.model, got, c.temperature)
		}
		if got := llm.Supports(KNOBEFFORT); got != c.effort {
			t.Errorf("%s Supports(effort) = %v, want %v", c.model, got, c.effort)
		}
	}
}

func TestSupportsUnknownKnob(t *testing.T) {
	llm, _ := NewProvider(PROVIDEROPENAI, "k", "gpt-4o")
	if llm.Supports("top_k") {
		t.Error("an unrecognised knob should not be reported as supported")
	}
}

// Supports must answer for the model the provider will actually use, which
// for an empty model is that provider's default.
func TestSupportsUsesProviderDefaultModel(t *testing.T) {
	var c Claude
	if c.Supports(KNOBTEMPERATURE) {
		t.Errorf("an unset model should fall back to %s, which rejects temperature", MODELCLAUDE)
	}
	if !c.Supports(KNOBEFFORT) {
		t.Errorf("an unset model should fall back to %s, which accepts effort", MODELCLAUDE)
	}

	var o OpenAI
	if !o.Supports(KNOBTEMPERATURE) {
		t.Errorf("an unset model should fall back to %s, which accepts temperature", MODELGPT35)
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

// The whole point of the capability check: a knob the model rejects must be an
// error at startup, not a 400 on the first query.
func TestApplyKnobsRejectsUnsupported(t *testing.T) {
	temp := 0.7

	claude, _ := NewProvider(PROVIDERANTHROPIC, "k", "claude-opus-5")
	if err := ApplyKnobs(claude, "claude-opus-5", &temp, ""); err == nil {
		t.Error("expected an error setting temperature on claude-opus-5")
	}

	openai, _ := NewProvider(PROVIDEROPENAI, "k", "gpt-3.5-turbo")
	if err := ApplyKnobs(openai, "gpt-3.5-turbo", nil, EFFORTHIGH); err == nil {
		t.Error("expected an error setting effort on gpt-3.5-turbo")
	}
}

func TestApplyKnobsRejectsInvalidEffort(t *testing.T) {
	llm, _ := NewProvider(PROVIDERANTHROPIC, "k", "claude-opus-5")
	if err := ApplyKnobs(llm, "claude-opus-5", nil, "turbo"); err == nil {
		t.Error("expected an error for an effort level that is not one of the EFFORT* constants")
	}
}

func TestApplyKnobsAcceptsSupported(t *testing.T) {
	temp := 0.7

	claude, _ := NewProvider(PROVIDERANTHROPIC, "k", "claude-opus-5")
	if err := ApplyKnobs(claude, "claude-opus-5", nil, EFFORTMEDIUM); err != nil {
		t.Fatalf("effort on claude-opus-5 should be accepted: %v", err)
	}
	if got := claude.(*Claude).effort; got != EFFORTMEDIUM {
		t.Errorf("effort = %q, want %q", got, EFFORTMEDIUM)
	}

	openai, _ := NewProvider(PROVIDEROPENAI, "k", "gpt-4o")
	if err := ApplyKnobs(openai, "gpt-4o", &temp, ""); err != nil {
		t.Fatalf("temperature on gpt-4o should be accepted: %v", err)
	}
	if got := openai.(*OpenAI).temperature; got == nil || *got != temp {
		t.Errorf("temperature = %v, want %v", got, temp)
	}
}

// An unset knob must stay unset rather than being defaulted to a zero value,
// which is why the config field is a pointer.
func TestApplyKnobsLeavesUnsetKnobsAlone(t *testing.T) {
	llm, _ := NewProvider(PROVIDEROPENAI, "k", "gpt-4o")
	if err := ApplyKnobs(llm, "gpt-4o", nil, ""); err != nil {
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
	if err := ApplyKnobs(llm, "gpt-4o", &zero, ""); err != nil {
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
