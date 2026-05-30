import unittest

from scripts.publish_to_notion import (
    extract_concise_note,
    markdown_heading_to_title,
    markdown_to_blocks,
    normalize_notion_page_id,
)


class PublishToNotionTests(unittest.TestCase):
    def test_normalize_notion_page_id_from_url(self) -> None:
        page_id = normalize_notion_page_id(
            "https://app.notion.com/p/"
            "Paper-Notes-1234567890abcdef1234567890abcdef?source=copy_link"
        )
        self.assertEqual(page_id, "12345678-90ab-cdef-1234-567890abcdef")

    def test_extract_concise_note_section(self) -> None:
        markdown = "# Paper\n\n## Notion concise note\n- One\n- Two\n\n## Limitations\nNone"
        self.assertEqual(extract_concise_note(markdown), "- One\n- Two")

    def test_markdown_to_blocks_handles_headings_and_bullets(self) -> None:
        blocks = markdown_to_blocks("# Title\n\n- Finding\n\nParagraph text")
        self.assertEqual(
            [block["type"] for block in blocks],
            ["heading_1", "bulleted_list_item", "paragraph"],
        )

    def test_markdown_heading_to_title_falls_back(self) -> None:
        self.assertEqual(markdown_heading_to_title("No heading", "fallback"), "fallback")


if __name__ == "__main__":
    unittest.main()
