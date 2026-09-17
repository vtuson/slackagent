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
