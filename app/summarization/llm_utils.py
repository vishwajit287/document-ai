
from app.llm.llm_factory import get_llm
from app.llm.usage_tracker import UsageTracker

from app.summarization.schemas import DocumentSummary


class SummarizationEngine:

    def __init__(self):

        self.llm = get_llm()

        self.structured_llm = (
            self.llm.with_structured_output(
                DocumentSummary,
                method="json_schema",
                include_raw=True
            )
        )

        self.tracker = UsageTracker()

    def invoke_text(self, prompt: str) -> str:

        start = self.tracker.start_timer()

        response = self.llm.invoke(prompt)

        self.tracker.stop_timer(start)

        self.tracker.add_usage(response)

        return response.content

    def invoke_structured(
        self,
        prompt: str
    ) -> DocumentSummary:

        start = self.tracker.start_timer()

        response = self.structured_llm.invoke(prompt)

        self.tracker.stop_timer(start)

        self.tracker.add_usage(response["raw"])

        if response["parsing_error"]:
            raise ValueError(
                f"Structured output parsing failed: "
                f"{response['parsing_error']}"
            )

        if response["parsed"] is None:
            raise ValueError(
                "The model did not return a valid summary."
            )

        return response["parsed"]

    def get_usage(self):

        return self.tracker.get_metrics()
