import os

import dspy
from dspy.teleprompt import BootstrapFewShot

# lm config
lm = dspy.LM(model="groq/llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
dspy.settings.configure(lm=lm)


class GenerateSearchQueires(dspy.Signature):
    """DSPY optimisation.
    Refine a claim into a list of search queries to verify its veracity."""

    claim: str = dspy.InputField(desc="The raw claim to be fact-checked.")  # pyright: ignore[reportInvalidTypeForm]
    context: str = dspy.InputField(  # pyright: ignore[reportInvalidTypeForm]
        desc="Previous search results to inform the query refinement."
    )
    queries: str = dspy.OutputField(
        desc="Exactly 3 distinct search queries to verify the claim, seperated by a semicolon. Do not include numbering."
    )  # pyright: ignore[reportInvalidTypeForm]


# program definition
class QueryGenerator(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(GenerateSearchQueires)

    def forward(self, claim, context=""):
        return self.generate(claim=claim, context=context)


# learning examples
# todo move this to file??
examples = [
    dspy.Example(
        claim="The Great Wall of China is visible from space.",
        context="",
        queries="Is the Great Wall of China visible from space; astronaut visibility Earth landmarks space; Great Wall of China myths debunked",
    ).with_inputs("claim", "context"),
    dspy.Example(
        claim="Drinking coffee stunts your growth.",
        context="",
        queries="does coffee stunt growth scientific evidence; caffeine effect on bone development children; coffee growth myth medical research",
    ).with_inputs("claim", "context"),
    dspy.Example(
        claim="Lightning never strikes the same place twice.",
        context="",
        queries="can lightning strike same place twice; lightning strike frequency tall structures; lightning rod multiple strikes evidence",
    ).with_inputs("claim", "context"),
    dspy.Example(
        claim="Humans only use 10 percent of their brains.",
        context="",
        queries="10 percent brain myth neuroscience; how much of human brain is used; brain activity research full brain usage",
    ).with_inputs("claim", "context"),
    dspy.Example(
        claim="Napoleon Bonaparte was very short.",
        context="",
        queries="Napoleon Bonaparte actual height historical record; Napoleon height myth origin; average height 18th century France comparison",
    ).with_inputs("claim", "context"),
]


# metric function
def query_quality_metric(example, prediction, trace=None):
    queries_output = prediction.queries

    # output must be a non-empty string
    if not isinstance(queries_output, str) or not queries_output.strip():
        return False

    # must contain exactly 3 colon-seperated queries
    parts = [q.strip() for q in queries_output.split(";") if q.strip()]
    if len(parts) != 3:
        return False

    # No two queries should share more than half their words
    for i, q1 in enumerate(parts):
        for q2 in parts[i + 1 :]:
            words1 = set(q1.lower().split())
            words2 = set(q2.lower().split())
            overlap = len(words1 & words2) / max(len(words1), len(words2))
            if overlap > 0.5:
                return False

    return True


if __name__ == "__main__":
    program = QueryGenerator()

    optimiser = BootstrapFewShot(
        metric=query_quality_metric, max_bootstrapped_demos=3, max_labeled_demos=4
    )

    compiled_program = optimiser.compile(program, trainset=examples)

    compiled_program.save("app/compiled_query_generator.json")
