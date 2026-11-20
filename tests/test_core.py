"""Fast tests for the logic that doesn't need a model (no downloads, runs in a second).

Run:  python -m unittest discover tests
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'eval'))

from build_eval_set import build, swap_letters  # noqa: E402
from core.faq_data import find_url, load_entries, load_pairs  # noqa: E402
from core.generation import clean_answer  # noqa: E402
from core.retrieval import normalize_question  # noqa: E402
from core.training import encode, with_short_names  # noqa: E402
from run_eval import token_f1  # noqa: E402


class DatasetTests(unittest.TestCase):
    def test_duplicates_are_merged_into_one_entry_per_question(self):
        entries = load_entries()
        self.assertEqual(len(entries), len({e.question for e in entries}))
        self.assertEqual(len(load_pairs()), 100)          # 172 raw pairs, 72 exact repeats
        self.assertEqual(sum(1 for e in entries if e.program), 56)

    def test_merged_entries_keep_the_most_detailed_answer(self):
        art_education = next(e for e in load_entries() if e.program == 'Art Education')
        self.assertIn('department', art_education.answer)

    def test_find_url_drops_trailing_punctuation(self):
        self.assertEqual(find_url('See https://www.pittstate.edu/x/index.html.'), 'https://www.pittstate.edu/x/index.html')
        self.assertIsNone(find_url('no link here'))


class RetrievalTextTests(unittest.TestCase):
    def test_university_name_is_removed_in_every_spelling(self):
        for question in ('Does Pittsburg State University offer online programs?',
                         'Does PSU offer online programs?', 'Does Pitt State offer online programs?'):
            self.assertEqual(normalize_question(question), 'Does offer online programs?')

    def test_name_inside_other_words_is_kept(self):
        self.assertEqual(normalize_question('PSUEDO topics'), 'PSUEDO topics')


class TrainingFormatTests(unittest.TestCase):
    def test_loss_only_covers_the_answer(self):
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(ROOT / 'core' / 'program_tokenizer')
        except OSError:
            self.skipTest('tokenizer not downloaded')
        ids, labels = encode(tokenizer, 'Question?', 'Answer.')
        question_len = len(tokenizer.encode('Question?' + tokenizer.eos_token))
        self.assertEqual(len(ids), len(labels))
        self.assertTrue(all(label == -100 for label in labels[:question_len]))   # question ignored
        self.assertEqual(labels[question_len:], ids[question_len:])               # answer scored
        self.assertEqual(ids[-1], tokenizer.eos_token_id)                         # answer ends the turn

    def test_short_name_copies_keep_the_answer(self):
        pairs = with_short_names([('What is the X program at Pittsburg State University?', 'A')])
        self.assertEqual({q for q, _ in pairs}, {'What is the X program at Pittsburg State University?',
                                                  'What is the X program at PSU?',
                                                  'What is the X program at Pitt State?'})
        self.assertTrue(all(a == 'A' for _, a in pairs))


class EvalTests(unittest.TestCase):
    def test_token_f1(self):
        self.assertEqual(token_f1('the art program', 'the art program'), 1.0)
        self.assertEqual(token_f1('nothing alike', 'the art program'), 0.0)
        self.assertAlmostEqual(token_f1('the art', 'the art program'), 0.8)

    def test_swap_letters_changes_the_longest_word(self):
        self.assertEqual(swap_letters('Chemistry'), 'Chemsitry')
        self.assertEqual(swap_letters('Art'), 'Art')          # too short to swap

    def test_every_kind_has_dev_and_test_questions(self):
        items = build()
        for kind in {i['kind'] for i in items}:
            splits = {i['split'] for i in items if i['kind'] == kind}
            self.assertEqual(splits, {'dev', 'test'}, kind)

    def test_eval_questions_are_not_training_questions(self):
        training = {q for q, _ in with_short_names(load_pairs())}
        leaked = [i['question'] for i in build() if i['question'] in training]
        self.assertEqual(leaked, [])

    def test_committed_eval_set_matches_the_builder(self):
        committed = json.loads((ROOT / 'eval' / 'questions.json').read_text())
        self.assertEqual(committed, build(), 'rerun: python eval/build_eval_set.py')


class GenerationTextTests(unittest.TestCase):
    def test_clean_answer(self):
        self.assertEqual(clean_answer('  . The Art program.'), 'The Art program.')


if __name__ == '__main__':
    unittest.main()
