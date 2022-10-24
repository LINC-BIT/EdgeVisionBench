from rich.markdown import Markdown, MarkdownElement, Paragraph, CodeBlock, Heading, BlockQuote, HorizontalRule, ListElement, ListItem, ImageItem
from rich.console import Console, ConsoleOptions, RenderResult
from typing import Any, ClassVar, Dict, List, Optional, Type, Union
from rich.text import Text
from rich.panel import Panel
from rich import box


class MyHeading(Heading):
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        text = self.text
        text.justify = "left"
        if self.level == 1:
            # Draw a border around h1s
            yield Panel(
                text,
                box=box.DOUBLE,
                style="markdown.h1.border",
            )
        else:
            # Styled text for h2 and beyond
            if self.level == 2:
                yield Text("")
            yield text


class MyMarkdown(Markdown):
    elements: ClassVar[Dict[str, Type[MarkdownElement]]] = {
        "paragraph": Paragraph,
        "heading": MyHeading,
        "code_block": CodeBlock,
        "block_quote": BlockQuote,
        "thematic_break": HorizontalRule,
        "list": ListElement,
        "item": ListItem,
        "image": ImageItem,
    }
