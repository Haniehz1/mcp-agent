# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "beautifulsoup4",
#     "pydantic",
#     "rich",
#     "typer",
# ]
# ///

import locale
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field
import json
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track
from pathlib import Path

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

app = typer.Typer()
console = Console()


class ModelBenchmarks(BaseModel):
    """
    Performance benchmarks for comparing different models.
    """

    __pydantic_extra__: dict[str, float] = Field(
        init=False
    )  # Enforces that extra fields are floats

    quality_score: float | None = None
    """A blended quality score for the model."""

    mmlu_score: float | None = None
    gsm8k_score: float | None = None
    bbh_score: float | None = None

    model_config = ConfigDict(extra="allow")


class ModelLatency(BaseModel):
    """
    Latency benchmarks for comparing different models.
    """

    time_to_first_token_ms: float = Field(gt=0)
    """ 
    Median Time to first token in milliseconds.
    """

    tokens_per_second: float = Field(gt=0)
    """
    Median output tokens per second.
    """


class ModelCost(BaseModel):
    """
    Cost benchmarks for comparing different models.
    """

    blended_cost_per_1m: float | None = None
    """
    Blended cost mixing input/output cost per 1M tokens.
    """

    input_cost_per_1m: float | None = None
    """
    Cost per 1M input tokens.
    """

    output_cost_per_1m: float | None = None
    """
    Cost per 1M output tokens.
    """

    model_config = ConfigDict(extra="allow")


class ModelMetrics(BaseModel):
    """
    Model metrics for comparing different models.
    """

    cost: ModelCost
    speed: ModelLatency
    intelligence: ModelBenchmarks


class ModelInfo(BaseModel):
    name: str
    description: str | None = None
    provider: str
    context_window: int | None = None
    tool_calling: bool | None = None
    structured_outputs: bool | None = None
    metrics: ModelMetrics

    model_config = ConfigDict(extra="allow")


def parse_context_window(context_str: str) -> int | None:
    """Parse context window strings like '131k', '1m', '128000' to integers."""
    if not context_str:
        return None

    context_str = context_str.strip().lower()
    try:
        # Handle k suffix (thousands)
        if context_str.endswith("k"):
            return int(float(context_str[:-1]) * 1000)
        # Handle m suffix (millions)
        elif context_str.endswith("m"):
            return int(float(context_str[:-1]) * 1000000)
        # Handle plain numbers
        else:
            return int(context_str.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def parse_html_to_models(html_content: str) -> list[ModelInfo]:
    soup = BeautifulSoup(html_content, "html.parser")
    models = []

    headers = [th.get_text(strip=True) for th in soup.find_all("th")]
    console.print(f"[bold blue]Found {len(headers)} headers[/bold blue]")

    # Cell index to header mapping:
    # 0: API Provider
    # 1: Model
    # 2: ContextWindow
    # 3: Function Calling
    # 4: JSON Mode
    # 5: License
    # 6: OpenAI Compatible
    # 7: API ID
    # 8: Footnotes
    # 9: Artificial AnalysisIntelligence Index
    # 10: MMLU-Pro (Reasoning & Knowledge)
    # 11: GPQA Diamond (Scientific Reasoning)
    # 12: Humanity's Last Exam (Reasoning & Knowledge)
    # 13: LiveCodeBench (Coding)
    # 14: SciCode (Coding)
    # 15: HumanEval (Coding)
    # 16: MATH-500 (Quantitative Reasoning)
    # 17: AIME 2024 (Competition Math)
    # 18: Chatbot Arena
    # 19: BlendedUSD/1M Tokens
    # 20: Input PriceUSD/1M Tokens
    # 21: Output PriceUSD/1M Tokens
    # 22: MedianTokens/s
    # 23: P5Tokens/s
    # 24: P25Tokens/s
    # 25: P75Tokens/s
    # 26: P95Tokens/s
    # 27: MedianFirst Chunk (s)
    # 28: First AnswerToken (s)
    # 29: P5First Chunk (s)
    # 30: P25First Chunk (s)
    # 31: P75First Chunk (s)
    # 32: P95First Chunk (s)
    # 33: TotalResponse (s)
    # 34: ReasoningTime (s)
    # 35: FurtherAnalysis

    # Find all table rows
    rows = soup.find_all("tr")[2:]  # Skip header rows

    console.print(f"[bold green]Processing {len(rows)} models...[/bold green]")

    for row in track(rows, description="Parsing models..."):
        cells = row.find_all("td")
        if not cells:  # Ensure we have enough cells
            continue

        try:
            # Extract provider from cells[0] (API Provider)
            provider_img = cells[0].find("img")
            provider = (
                provider_img["alt"].replace(" logo", "") if provider_img else "Unknown"
            )

            # Extract model name from cells[1] (Model)
            model_name_elem = cells[1].find("span")
            if model_name_elem:
                display_name = model_name_elem.text.strip()
            else:
                display_name = cells[1].get_text(strip=True)

            # Extract API ID from cells[7] (API ID)
            api_id_text = cells[7].get_text(strip=True)
            if api_id_text:
                api_id = api_id_text
            else:
                # Use model name as fallback
                api_id = (
                    display_name.lower()
                    .replace(" ", "-")
                    .replace("(", "")
                    .replace(")", "")
                )

            # Extract context window from cells[2] (ContextWindow)
            context_window_text = cells[2].get_text(strip=True)
            context_window = parse_context_window(context_window_text)

            # Extract tool calling from cells[3] (Function Calling)
            # Check for checkmark or X icon
            tool_calling_elem = cells[3].find("svg")
            if tool_calling_elem:
                # Check if it has "Yes" in the title
                title_elem = tool_calling_elem.find("title")
                tool_calling = (
                    True if title_elem and "Yes" in title_elem.text else False
                )
            else:
                tool_calling = None

            # Extract structured outputs from cells[4] (JSON Mode)
            structured_outputs_elem = cells[4].find("svg")
            if structured_outputs_elem:
                # Check if it has "Yes" in the title
                title_elem = structured_outputs_elem.find("title")
                structured_outputs = (
                    True if title_elem and "Yes" in title_elem.text else False
                )
            else:
                structured_outputs = None

            # Extract quality score from cells[9] (Artificial AnalysisIntelligence Index)
            quality_text = cells[9].get_text(strip=True).replace("%", "")
            quality_score = (
                locale.atof(quality_text)
                if quality_text.replace(".", "").isdigit()
                else 0
            )

            # Extract cost from cells[19] (BlendedUSD/1M Tokens)
            cost_text = cells[19].get_text(strip=True).replace("$", "").replace(",", "")
            blended_cost = locale.atof(cost_text) if cost_text else 0

            # Extract performance metrics
            # Tokens per second from cells[22] (MedianTokens/s)
            tokens_text = cells[22].get_text(strip=True).replace(",", "")
            tokens_per_sec = locale.atof(tokens_text) if tokens_text else 0.1

            # Latency from cells[27] (MedianFirst Chunk (s))
            latency_text = cells[27].get_text(strip=True).replace(",", "")
            latency_sec = locale.atof(latency_text) if latency_text else 0

            model_info = ModelInfo(
                name=api_id,
                description=display_name,
                provider=provider,
                context_window=context_window,
                tool_calling=tool_calling,
                structured_outputs=structured_outputs,
                metrics=ModelMetrics(
                    cost=ModelCost(blended_cost_per_1m=blended_cost),
                    speed=ModelLatency(
                        time_to_first_token_ms=latency_sec
                        * 1000,  # Convert to milliseconds
                        tokens_per_second=tokens_per_sec,
                    ),
                    intelligence=ModelBenchmarks(quality_score=quality_score),
                ),
            )

            models.append(model_info)

        except Exception as e:
            console.print(f"[red]Error processing row: {e}[/red]")
            console.print(f"[yellow]Row content: {str(row)}[/yellow]")
            continue

    return models


def export_to_json(
    models: list[ModelInfo], output_file: str = "model_benchmarks5.json"
):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([m.model_dump() for m in models], f, indent=2)


def display_summary(models: list[ModelInfo]):
    """Display a summary table of parsed models."""
    table = Table(title=f"Parsed Models Summary ({len(models)} models)")

    table.add_column("#", style="dim", width=3)
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Model", style="magenta", max_width=50)
    table.add_column("Context", justify="right", style="green")
    table.add_column("Tools", justify="center")
    table.add_column("JSON", justify="center")
    table.add_column("Quality", justify="right", style="yellow")
    table.add_column("Cost/1M", justify="right", style="red")
    table.add_column("Speed", justify="right", style="blue")

    for idx, model in enumerate(models, 1):
        # Truncate long model names
        model_name = model.description or model.name
        if len(model_name) > 50:
            model_name = model_name[:47] + "..."

        table.add_row(
            str(idx),
            model.provider,
            model_name,
            f"{model.context_window:,}" if model.context_window else "N/A",
            "✓" if model.tool_calling else "✗" if model.tool_calling is False else "?",
            "✓"
            if model.structured_outputs
            else "✗"
            if model.structured_outputs is False
            else "?",
            f"{model.metrics.intelligence.quality_score:.1f}%"
            if model.metrics.intelligence.quality_score
            else "N/A",
            f"${model.metrics.cost.blended_cost_per_1m:.2f}"
            if model.metrics.cost.blended_cost_per_1m
            else "N/A",
            f"{model.metrics.speed.tokens_per_second:.0f} t/s"
            if model.metrics.speed.tokens_per_second
            else "N/A",
        )

    console.print(table)


@app.command()
def main(
    input_file: Path = typer.Argument(
        ...,
        help="Path to the HTML file containing the benchmark table",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    output_file: Path = typer.Argument(
        "src/mcp_agent/data/artificial_analysis_llm_benchmarks.json",
        help="Path to the output JSON file",
        resolve_path=True,
    ),
):
    """
    Parse LLM benchmark HTML tables from Artificial Analysis and convert to JSON.
    """
    console.print(f"[bold]Reading HTML from:[/bold] {input_file}")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            html_content = f.read()

        models = parse_html_to_models(html_content)

        if not models:
            console.print("[red]No models found in the HTML file![/red]")
            raise typer.Exit(1)

        console.print(
            f"\n[bold green]Successfully parsed {len(models)} models![/bold green]\n"
        )

        display_summary(models)

        export_to_json(models, str(output_file))
        console.print(f"\n[bold]Output saved to:[/bold] {output_file}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
