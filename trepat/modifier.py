import torch

# lambo calls torch.serialization.add_safe_globals(), which only exists on
# torch>=2.4. This repo is pinned to torch 2.1.2, where torch.load() had no
# allowlist mechanism and unpickled unconditionally, so a no-op shim here
# preserves that behavior instead of crashing on import.
if not hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals = lambda *args, **kwargs: None

from lambo.segmenter.lambo import Lambo
from configuration import MAX_CHANGE_TOTAL, MAX_CHANGE_FRAGMENT, MAX_VARIANTS, MIN_CHUNK_OR_SENTENCE_LENGTH
import numpy as np
import random

DIRECTION_STOP = 0
DIRECTION_UP = 1
DIRECTION_LEFT = 2
DIRECTION_UPLEFT = 3

class Modifier:
    def __init__(self, rephraser, splitter="sentences", weak=False, max_variants=MAX_VARIANTS):
        self.rephraser = rephraser
        self.lambo = Lambo.get('English')
        self.original_text = None
        self.best_variants = []
        self.tested_variants = []
        self.untested_variants = []
        self.current_variant = None
        self.max_variants = max_variants
        self.changes = []
        self.variant_counter = 0
        self.default_splitter = splitter
        self.last_variant = None
        self.generated_variants = None
        self.weak = weak
    
    def init(self, text):
        self.variant_counter = 0
        self.last_variant = ''
        self.generated_variants = set()
        self.reset(text)
    
    def reset(self, text):
        # print("Resetting modifier with text: " + text)
        self.original_text = text
        # print("Getting the initial changes...")
        self.changes = self.get_changes_everywhere(self.original_text)
        # print("Found " + str(len(self.changes)))
        self.best_variants = []
        self.tested_variants = []
        self.untested_variants = self.explore_initial_variants(self.original_text, self.changes)
        self.current_variant = None
    
    def split(self, text, splitter=None):
        if not splitter:
            splitter = self.default_splitter
        if splitter == "sentences":
            return self.split_on_sentences(text)
        elif splitter == "newlines":
            return self.split_on_newlines(text)
        elif splitter == "paired":
            return self.split_on_pairs(text)
        elif splitter == "chunks":
            return self.split_on_separators(text)
        elif splitter == "cascade":
            return self.split_cascade(text)
        elif splitter == "none":
            return [(0, text)]
        else:
            assert False
    
    def split_on_sentences(self, text):
        document = self.lambo.segment(text)
        offset = 0
        result = []
        for turn in document.turns:
            for sentence in turn.sentences:
                if len(result) > 0 and (len(sentence.text) < MIN_CHUNK_OR_SENTENCE_LENGTH or len(
                        result[-1][1]) < MIN_CHUNK_OR_SENTENCE_LENGTH):
                    result[-1] = (result[-1][0], result[-1][1] + sentence.text)
                else:
                    result.append((offset, sentence.text))
                offset += len(sentence.text)
        return result
    
    @staticmethod
    def split_on_newlines(text):
        parts = text.split("\n")
        offset = 0
        result = []
        for part in parts:
            if len(part.strip()) > 1:
                result.append((offset, part))
            offset += (len(part) + 1)
        return result
    
    @staticmethod
    def split_on_pairs(text):
        separator = " ~ "
        parts = text.split(separator)
        offset = 0
        result = []
        for part in parts:
            result.append((offset, part))
            offset += len(part) + len(separator)
        return result
    
    def split_on_token_chunks(self, text):
        CHUNK_SIZE = 5
        tokenised = self.rephraser.tokenizer(text, return_offsets_mapping=True)
        offsets = [(begin, end) for (begin, end) in tokenised['offset_mapping'] if begin != end]
        result = []
        while len(offsets) > 0:
            chunk_length = min(CHUNK_SIZE, len(offsets))
            result.append((offsets[0][0], text[(offsets[0][0]):(offsets[chunk_length - 1][1])]))
            offsets = offsets[chunk_length:]
        return result
    
    @staticmethod
    def split_on_separators(text):
        separators = [', ', ' - ', ' — ', ' – ', '\' ', ' \'', '\" ', ' \"', ' “', '” ', ' ‘', '’ ', ': ']
        result = [(0, text)]
        result_new = []
        for separator in separators:
            for offset, chunk_text in result:
                parts = chunk_text.split(separator)
                if len(parts) == 1:
                    result_new.append((offset, chunk_text))
                    continue
                offset_new = offset
                for part in parts:
                    if len(result_new) > 0 and (len(part) < MIN_CHUNK_OR_SENTENCE_LENGTH or len(
                            result_new[-1][1]) < MIN_CHUNK_OR_SENTENCE_LENGTH):
                        concatenated_chunk = text[(result_new[-1][0]):(offset_new + len(part))]
                        result_new[-1] = (result_new[-1][0], concatenated_chunk)
                    else:
                        result_new.append((offset_new, part))
                    assert (result_new[-1][1] == text[(result_new[-1][0]):(result_new[-1][0] + len(result_new[-1][1]))])
                    offset_new += len(part) + len(separator)
            result = result_new
            result_new = []
        return result
    
    def split_cascade(self, text):
        result = [(0, text)]
        for splitter in ['paired', 'newlines', 'sentences', 'chunks']:
            result_new = []
            for old_offset, old_fragment in result:
                new_split = self.split(old_fragment, splitter)
                result_new.extend([(old_offset + offset, fragment) for offset, fragment in new_split])
            result = result_new
        return result
    
    def get_changes_everywhere(self, text):
        fragments = self.split(text)
        result = set()
        for offset, fragment_text in fragments:
            all_changes = self.get_changes_in_fragment(fragment_text)
            all_changes = [change for change in all_changes if change[1] not in ['’', '‘', '“', '”']]
            if self.weak:
                limited_changes = all_changes
            else:
                limited_changes = [change for change in all_changes if
                                   (max(len(change[1]), len(change[2])) < MAX_CHANGE_TOTAL * len(text)) and (
                                           max(len(change[1]), len(change[2])) < MAX_CHANGE_FRAGMENT * len(fragment_text))]
            for change in limited_changes:
                result.add(((change[0][0] + offset, change[0][1] + offset), change[1], change[2]))
        return result
    
    def get_changes_in_fragment(self, text):
        # print("Running fragment rephraser...")
        rephrasings = self.rephraser.rephrase(text)
        all_changes = set()
        # print("Collecting changes...")
        for rephrasing in rephrasings:
            if self.weak:
                all_changes.add(((0, len(text)), text, rephrasing))
                continue
            document_o = self.lambo.segment(text)
            document_c = self.lambo.segment(rephrasing)
            tokens_o = []
            tokens_c = []
            for turn in document_o.turns:
                for sentence in turn.sentences:
                    for token in sentence.tokens:
                        tokens_o.append(token)
            for turn in document_c.turns:
                for sentence in turn.sentences:
                    for token in sentence.tokens:
                        tokens_c.append(token)
            distances = np.zeros((len(tokens_o) + 1, len(tokens_c) + 1))
            path = np.zeros((len(tokens_o) + 1, len(tokens_c) + 1))
            distances[0, :] = range(distances.shape[1])
            distances[:, 0] = range(distances.shape[0])
            path[0, :] = [DIRECTION_LEFT] * distances.shape[1]
            path[:, 0] = [DIRECTION_UP] * distances.shape[0]
            path[0, 0] = DIRECTION_STOP
            for i_o in range(1, len(tokens_o) + 1):
                for i_c in range(1, len(tokens_c) + 1):
                    replacement_tax = 0 if tokens_o[i_o - 1].text == tokens_c[i_c - 1].text else 1
                    adding_tax = 1
                    removing_tax = 1
                    replacement_dist = distances[i_o - 1, i_c - 1] + replacement_tax
                    adding_dist = distances[i_o, i_c - 1] + adding_tax
                    removing_dist = distances[i_o - 1, i_c] + removing_tax
                    best_dist = min(replacement_dist, adding_dist, removing_dist)
                    distances[i_o, i_c] = best_dist
                    if best_dist == replacement_dist:
                        path[i_o, i_c] = DIRECTION_UPLEFT
                    elif best_dist == adding_dist:
                        path[i_o, i_c] = DIRECTION_LEFT
                    elif best_dist == removing_dist:
                        path[i_o, i_c] = DIRECTION_UP
                    else:
                        assert (False)
            i_o = len(tokens_o)
            i_c = len(tokens_c)
            operations = []
            while path[i_o, i_c] != DIRECTION_STOP:
                if path[i_o, i_c] == DIRECTION_UPLEFT:
                    operation = (
                        "KEEP" if tokens_o[i_o - 1].text == tokens_c[i_c - 1].text else "REPLACE",
                        tokens_o[i_o - 1],
                        tokens_c[i_c - 1])
                    i_o -= 1
                    i_c -= 1
                elif path[i_o, i_c] == DIRECTION_LEFT:
                    operation = ("ADD", None, tokens_c[i_c - 1])
                    i_c -= 1
                elif path[i_o, i_c] == DIRECTION_UP:
                    operation = ("REMOVE", tokens_o[i_o - 1], None)
                    i_o -= 1
                else:
                    assert False
                operations = [operation] + operations
            operations_chained = []
            current_operation = ([], [])
            for operation_type, token_o, token_c in operations:
                if operation_type == "REPLACE":
                    current_operation[0].append(token_o)
                    current_operation[1].append(token_c)
                elif operation_type == "ADD":
                    current_operation[1].append(token_c)
                elif operation_type == "REMOVE":
                    current_operation[0].append(token_o)
                elif operation_type == "KEEP":
                    if current_operation != ([], []):
                        operations_chained.append(current_operation)
                        current_operation = ([], [])
                else:
                    assert False
            if current_operation != ([], []):
                operations_chained.append(current_operation)
            for tokens_o, tokens_c in operations_chained:
                if len(tokens_o) == 0 or len(tokens_c) == 0:
                    # Note: we ignore additions and removals, only taking replacements
                    continue
                all_changes.add(((tokens_o[0].begin, tokens_o[-1].end),
                                 text[tokens_o[0].begin:tokens_o[-1].end],
                                 rephrasing[tokens_c[0].begin:tokens_c[-1].end]))
        return all_changes
    
    def explore_initial_variants(self, text, changes):
        # print("Exploring the possible variants...")
        result = []
        for change in changes:
            variant = Variant(text)
            variant.apply_change(change)
            if variant.current_text in self.generated_variants:
                continue
            result.append(variant)
            self.generated_variants.add(variant.current_text)
        # print("Found " + str(len(result)) + " variants, sorting.")
        result = self.sort_variants(result)
        return result
    
    def get_next_variant(self):
        if self.variant_counter > self.max_variants:
            # print("Max variants reached, failing. ")
            return None
        else:
            self.variant_counter += 1
        
        if len(self.untested_variants) > 0:
            # print(str(self.variant_counter) + " Asked for a variant, providing: ")
            self.current_variant = self.untested_variants[0]
            self.untested_variants = self.untested_variants[1:]
            # print("--> " + self.current_variant.current_text)
            return self.current_variant.current_text
        else:
            # print(
            #    str(self.variant_counter) + " Asked for a variant, but nothing left. Sorting tested variants and selecting the best.")
            self.tested_variants = sorted(self.tested_variants, key=lambda variant: - variant.value)
            self.best_variants = self.tested_variants[:5]
            for variant in self.best_variants:
                for change in self.changes:
                    if not variant.has_change(change):
                        new_variant = variant.duplicate()
                        new_variant.apply_change(change)
                        if new_variant.current_text in self.generated_variants:
                            continue
                        self.untested_variants.append(new_variant)
                        self.generated_variants.add(new_variant.current_text)
            if len(self.untested_variants) > 0:
                self.untested_variants = self.sort_variants(self.untested_variants)
                # print("Obtained " + str(len(self.untested_variants)))
                self.current_variant = self.untested_variants[0]
                self.untested_variants = self.untested_variants[1:]
                # print("--> " + self.current_variant.current_text)
                return self.current_variant.current_text
            else:
                # print("Nothing useful found, resetting with best variant.")
                if len(self.best_variants) == 0:
                    # print("Failed.")
                    return None
                else:
                    self.reset(self.best_variants[0].current_text)
                    return self.get_next_variant()
    
    @staticmethod
    def sort_variants(variants):
        result = sorted(variants, key=lambda variant: np.sum(
            [max(len(old_text), len(new_text)) * max(len(old_text) * 1.0 / (len(new_text)+0.1),
                                                     len(new_text) * 1.0 / (len(old_text)+0.1)) for
             _, old_text, new_text in variant.changes]))
        return result
    
    def get_feedback(self, value):
        if self.weak:
            # print("Weak mode, generating random feedback.")
            value = random.random()
        # print("Obtained feedback of " + str(value) + ", saving.")
        self.current_variant.get_feedback(value)
        self.tested_variants.append(self.current_variant)
        self.current_variant = None


class Variant:
    def __init__(self, text):
        self.original_text = text
        self.current_text = text
        self.changes = []
        self.value = -1
    
    def apply_change(self, change):
        (begin, end), old_text, new_text = change
        adjusted_begin = begin
        adjusted_end = end
        for (previous_begin, previous_end), previous_old_text, previous_new_text in self.changes:
            if previous_end <= begin:
                # if the change already made happened before, adjust the coordinates
                offset = len(previous_new_text) - len(previous_old_text)
                adjusted_begin += offset
                adjusted_end += offset
        self.changes.append(change)
        self.current_text = self.current_text[:adjusted_begin] + new_text + self.current_text[adjusted_end:]
    
    def get_feedback(self, value):
        self.value = value
    
    def duplicate(self):
        result = Variant(self.original_text)
        for (begin, end), old_text, new_text in self.changes:
            new_change = ((begin, end), old_text, new_text)
            result.apply_change(new_change)
        return result
    
    def has_change(self, change):
        (begin, end), old_text, new_text = change
        for old_change in self.changes:
            (begin_o, end_o), old_text_o, new_text_o = old_change
            if begin_o == begin and end_o == end and old_text_o == old_text and new_text_o == new_text:
                return True
            if (begin_o >= begin and begin_o < end) or (end_o > begin and end_o <= end) or (
                    begin >= begin_o and begin < end_o) or (end > begin_o and end <= end_o):
                return True
        return False
    
    def add(self, text):
        pass
