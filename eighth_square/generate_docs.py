import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import markdown

from algorithms.registry import ALGORITHMS, PAPERS
from environments.registry import ENVIRONMENTS

HERE = os.path.dirname(os.path.abspath(__file__))

CSS = """body { font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 950px; margin: 40px auto; padding: 0 24px; line-height: 1.55; color: #1a1a1a; }
pre { background: #f6f8fa; padding: 14px; border-radius: 6px; overflow-x: auto; font-size: 13px; }
code { background: #f6f8fa; padding: 1px 4px; border-radius: 3px; font-size: 13px; }
pre code { background: none; padding: 0; }
h1 { border-bottom: 2px solid #eaeef2; padding-bottom: 8px; }
h2 { margin-top: 40px; border-bottom: 1px solid #eaeef2; padding-bottom: 4px; }
h3 { margin-top: 28px; }
table { border-collapse: collapse; margin: 14px 0; width: 100%; font-size: 14px; }
th, td { border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; }
th { background: #f6f8fa; }
a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 22px 0; }
blockquote { border-left: 4px solid #d0d7de; margin: 0; padding: 2px 16px; color: #57606a; }
ul, ol { margin: 4px 0; }
"""

MATHJAX = """<script>
window.MathJax = {
  options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea'] },
  tex: {
    inlineMath: [['\\(', '\\)'], ['$', '$']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
"""


def to_html(title, markdown_text):
    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    return (
        f'<html><head><meta charset="utf-8"><title>{title}</title><style>\n{CSS}'
        f"</style>\n{MATHJAX}</head><body>{body}</body></html>"
    )


def md_escape(text):
    return str(text).replace("|", "\\|")


def generate_algorithms_md():
    lines = [
        "# Algorithm Table",
        "",
        "Every algorithm in the registry: what it is, where it comes from, and where it can run. "
        "`implemented` means you can train it today; `stub` means it is planned but not yet written.",
        "",
    ]
    for status in ("implemented", "stub"):
        lines += [
            f"## {status.capitalize()}",
            "",
            "| name | family | policy | action space | state space | source | paper | description | notes | compatible environments |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, a in sorted(ALGORITHMS.items()):
            if a["status"] != status:
                continue
            paper = PAPERS.get(name)
            paper_cell = f"[paper]({paper})" if paper else "—"
            lines.append(
                f"| {name} | {a['family']} | {a['policy']} | {a['action_space']} | {a['state_space']} "
                f"| {md_escape(a['source'])} | {paper_cell} | {md_escape(a['description'])} | {md_escape(a['notes'])} "
                f"| {', '.join(a['compatible_envs']) or '—'} |"
            )
        lines.append("")
    lines += ["## Use cases", ""]
    for name, a in sorted(ALGORITHMS.items()):
        lines.append(f"- **{name}**: {', '.join(a['use_cases'])}")
    lines.append("")
    path = os.path.join(HERE, "ALGORITHMS.html")
    with open(path, "w") as f:
        f.write(to_html("Algorithm Table", "\n".join(lines)))
    print(f"wrote {path}")


def generate_environments_md():
    lines = [
        "# Environment Table",
        "",
        "| name | gym id | action space | state space | family | description | use cases |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, e in sorted(ENVIRONMENTS.items()):
        lines.append(
            f"| {name} | {e['gym_id'] or 'custom'} | {e['action_space']} | {e['state_space']} "
            f"| {e['family']} | {md_escape(e['description'])} | {', '.join(e['use_cases'])} |"
        )
    lines.append("")
    lines += [
        "## Compatibility rules",
        "",
        "- An algorithm's `action_space` / `state_space` tags must match the environment's.",
        "- `both` accepts either kind.",
        "- The `compatible_envs` list in ALGORITHMS.html is the final word; "
        "`run_experiment.py` refuses mismatches with a clear error.",
    ]
    path = os.path.join(HERE, "ENVIRONMENTS.html")
    with open(path, "w") as f:
        f.write(to_html("Environment Table", "\n".join(lines)))
    print(f"wrote {path}")


if __name__ == "__main__":
    generate_algorithms_md()
    generate_environments_md()
