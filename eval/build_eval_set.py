"""Build eval/questions.json: questions the chatbot never saw in training, each with the answer it
should give (an entry of data/programs.json) or "none" when it should say it doesn't know.

Kinds of questions:
  paraphrase   - a per-program question in a new wording (templates not used in training)
  typo         - the same, with two letters swapped in the program name
  general      - a hand-written rewording of each general question (online, housing, ...)
  not_in_data  - programs the dataset doesn't cover (some PSU has, some it doesn't)
  off_topic    - nothing to do with PSU programs (two are close on purpose: cars, computers)

Within each kind, every third question goes to "dev" (used to choose settings such as the retrieval threshold),
the rest to "test" (used only for the final numbers). The split is fixed so results are repeatable.

Run:  python eval/build_eval_set.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.faq_data import load_entries  # noqa: E402

# New wordings; training only used "What is the X program at ..." and "Tell me about the X program at ..."
TEMPLATES = [
    'Is there a {p} program at PSU?',
    'Can I study {p} at Pitt State?',
    'What can you tell me about {p} at Pittsburg State?',
    'info on the {p} program',
    'Does Pittsburg State have {p}?',
]

# One rewording per general question, keyed by the dataset's own question
GENERAL = {
    'Does Pittsburg State University have an Accounting MBA program?': 'Can I get an MBA in accounting at PSU?',
    'What is the Additive Manufacturing Certificate program?': 'Is there a certificate in 3D printing / additive manufacturing?',
    'Can I study Advanced Practice Nursing at Pittsburg State University?': 'Does PSU have a doctorate for nurse practitioners?',
    'What does the Advanced Studies in Leadership program at Pittsburg State University cover?': "What's in the advanced studies in leadership program?",
    'How many academic programs does Pittsburg State University offer?': 'How many programs does Pitt State have?',
    'What types of degrees are available at Pittsburg State University?': 'Which degree levels can I earn at PSU?',
    'Does Pittsburg State University offer online programs?': 'Can I take classes online at PSU?',
    'Can I study healthcare-related programs at Pittsburg State University?': 'Are there health care degrees at Pitt State?',
    'What are some of the engineering programs available at Pittsburg State University?': 'Which engineering degrees does PSU have?',
    'Are there any special emphasis programs at Pittsburg State University?': 'Does PSU offer emphases within majors?',
    'What certificate programs does Pittsburg State University offer?': 'What certificates can I get at PSU?',
    'What kind of business programs are available at Pittsburg State University?': 'What business degrees does Pitt State offer?',
    'How can I explore programs offered by Pittsburg State University?': 'Where can I see the full list of PSU programs?',
    'Does Pittsburg State University offer any programs in science and technology?': 'Are there STEM programs at PSU?',
    'Are there programs specifically for educators at Pittsburg State University?': 'Does PSU have programs for teachers?',
    "What is the focus of Pittsburg State University's programs?": "What do PSU's programs focus on?",
    'What are the admission requirements for Pittsburg State University?': 'What do I need to get admitted to PSU?',
    'Does Pittsburg State University offer financial aid?': 'Can I get scholarships or financial aid at Pitt State?',
    'What student organizations are available at Pittsburg State University?': 'What clubs can I join at PSU?',
    'What are the housing options at Pittsburg State University?': 'Where can students live at PSU?',
    'What career services are available at Pittsburg State University?': 'Does PSU help students find jobs?',
    'What is the campus life like at Pittsburg State University?': 'What is student life like at Pitt State?',
    'Does Pittsburg State University offer study abroad programs?': 'Can I study abroad through PSU?',
    'What support services are available for students at Pittsburg State University?': 'What student support does PSU offer?',
    'What are the library resources at Pittsburg State University?': "What's available at the PSU library?",
    'What athletic programs are available at Pittsburg State University?': 'What sports teams does Pitt State have?',
}

NOT_IN_DATA = [
    'Is there a dental school at PSU?',
    'Can I study law at Pittsburg State?',
    'Does PSU have a veterinary medicine program?',
    'Is there a pharmacy degree at Pitt State?',
    'Does Pittsburg State offer aerospace engineering?',
    'Is there a marine biology program at PSU?',
    'Does PSU have an architecture degree?',
    # Real PSU programs that the 2024 dataset doesn't include (it stops at the letter C)
    'Is there a Graphic Design program at PSU?',
    'Can I study Mathematics at Pitt State?',
    'Does Pittsburg State have Social Work?',
    'info on the Spanish program',
    'What can you tell me about Theatre at Pittsburg State?',
    'Is there a Plastics Engineering Technology program at PSU?',
    'Can I study History at Pitt State?',
]

OFF_TOPIC = [
    "What's the weather like in Kansas today?",
    'Tell me a joke',
    'Who won the Super Bowl?',
    'Write me a poem about autumn',
    'What is 2 + 2?',
    'What is the capital of France?',
    'Give me a recipe for pancakes',
    'Should I buy Tesla stock?',
    "How do I fix my car's brakes?",
    'Explain how a computer works',
]


def swap_letters(name):
    """Swap two letters in the middle of the longest word: 'Chemistry' -> 'Chemsitry'."""
    words = name.split()
    i = max(range(len(words)), key=lambda k: len(words[k]))
    word = words[i]
    if len(word) < 5:
        return name
    mid = len(word) // 2
    words[i] = word[:mid] + word[mid + 1] + word[mid] + word[mid + 2:]
    return ' '.join(words)


def build():
    entries = load_entries()
    missing = set(GENERAL) - {e.question for e in entries}
    if missing:
        raise SystemExit(f'GENERAL keys not in the dataset: {missing}')

    items = []
    programs = [e for e in entries if e.program]
    for n, entry in enumerate(programs):
        template = TEMPLATES[n % len(TEMPLATES)]
        items.append({'kind': 'paraphrase', 'question': template.format(p=entry.program),
                      'expected': entry.question})
        if n % 5 == 2:  # every fifth program also gets a typo version (12 of 56)
            # a different template than the clean question, so the typo isn't the only change
            typo = TEMPLATES[(n + 1) % len(TEMPLATES)].format(p=swap_letters(entry.program))
            items.append({'kind': 'typo', 'question': typo, 'expected': entry.question})
    for question in (e.question for e in entries if not e.program):
        items.append({'kind': 'general', 'question': GENERAL[question], 'expected': question})
    items += [{'kind': 'not_in_data', 'question': q, 'expected': 'none'} for q in NOT_IN_DATA]
    items += [{'kind': 'off_topic', 'question': q, 'expected': 'none'} for q in OFF_TOPIC]

    # Rotate within each kind, so every kind has questions in both dev and test
    seen_per_kind = {}
    for item in items:
        n = seen_per_kind.get(item['kind'], 0)
        seen_per_kind[item['kind']] = n + 1
        item['split'] = 'dev' if n % 3 == 0 else 'test'
    return items


if __name__ == '__main__':
    items = build()
    out = ROOT / 'eval' / 'questions.json'
    out.write_text(json.dumps(items, indent=2, ensure_ascii=False) + '\n')
    counts = {}
    for item in items:
        key = (item['split'], item['kind'])
        counts[key] = counts.get(key, 0) + 1
    print(f'{len(items)} questions -> {out.relative_to(ROOT)}')
    for (split, kind), n in sorted(counts.items()):
        print(f'  {split:4} {kind:12} {n}')
