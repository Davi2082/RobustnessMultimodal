import random
import sys

random.seed(10)

victims = ['BiLSTM', 'BERT', 'GEMMA', 'GEMMA7B']
tasks = ['PR2', 'FC', 'RD', 'HN']
scenario2baseline = {'PR2_BiLSTM': 'BERTattack', 'FC_BiLSTM': 'BERTattack', 'RD_BiLSTM': 'FastBERTattack',
                     'HN_BiLSTM': 'FastBERTattack', 'PR2_BERT': 'BERTattack', 'FC_BERT': 'BERTattack',
                     'RD_BERT': 'TextTrojaners', 'HN_BERT': 'FastBERTattack', 'PR2_GEMMA': 'FastBERTattack',
                     'FC_GEMMA': 'BERTattack', 'RD_GEMMA': 'FastBERTattack', 'HN_GEMMA': 'FastBERTattack',
                     'PR2_GEMMA7B': 'BERTattack', 'FC_GEMMA7B': 'BERTattack', 'RD_GEMMA7B': 'FastBERTattack',
                     'HN_GEMMA7B': 'FastBERTattack'}
path_attacks = sys.argv[1]
path_BODEGA = sys.argv[2]
targeted_variant = 'False'
attacker = 'TREPAT-OLMO7B'
question_no = {'PR2': 94, 'FC': 57, 'RD': 23, 'HN': 25}
annotators = ['A', 'B']
shared_part = 0.25

# Reading attacks
all_aes = {}
all_originalspans = {}
for task in tasks:
    for victim in victims:
        all_aes[task + '_' + victim] = {}
        all_originalspans[task + '_' + victim] = {}
        baseline = scenario2baseline[task + '_' + victim]
        for variant, method_name in {(attacker, 'TREPAT'), (baseline, 'baseline')}:
            for line in open(
                    path_attacks / ('raw_' + task + '_' + targeted_variant + '_' + variant + '_' + victim + '.tsv')):
                parts = line.strip().split('\t')
                original = parts[1].replace('>', '').replace('<', '')
                ae = parts[2]
                if original not in all_aes[task + '_' + victim]:
                    all_aes[task + '_' + victim][original] = {}
                    all_originalspans[task + '_' + victim][original] = {}
                all_aes[task + '_' + victim][original][method_name] = ae
                all_originalspans[task + '_' + victim][original][method_name] = parts[1]

# Combining the change ranges in originals from both methods
for task in tasks:
    for victim in victims:
        for original in all_originalspans[task + '_' + victim]:
            if len(all_originalspans[task + '_' + victim][original]) == 2:
                text1 = all_originalspans[task + '_' + victim][original]['TREPAT'].replace('<<', '--').replace('>>',
                                                                                                               '--')
                text2 = all_originalspans[task + '_' + victim][original]['baseline'].replace('<<', '--').replace('>>',
                                                                                                                 '--')
                combined = ''
                i_1 = 0
                i_2 = 0
                in_1 = False
                in_2 = False
                while True:
                    if text1[i_1] == '<' and not in_1:
                        if not in_2:
                            combined += '<'
                        in_1 = True
                        i_1 += 1
                    elif text1[i_1] == '>' and in_1:
                        in_1 = False
                        if not in_2:
                            combined += '>'
                        i_1 += 1
                    if text2[i_2] == '<' and not in_2:
                        in_2 = True
                        if not in_1:
                            combined += '<'
                        i_2 += 1
                    elif text2[i_2] == '>' and in_2:
                        in_2 = False
                        if not in_1:
                            combined += '>'
                        i_2 += 1
                    if (i_1 >= len(text1)) or (i_2 >= len(text2)):
                        break
                    # assert(text1[i_1]==text2[i_2])
                    combined = combined + text1[i_1]
                    i_1 += 1
                    i_2 += 1
                    if (i_1 >= len(text1)) or (i_2 >= len(text2)):
                        break
                all_originalspans[task + '_' + victim][original]['combined'] = combined

# Remembering newlines
newlines = {}
for task in tasks:
    newlines[task] = {}
    for line in open(path_BODEGA / task / 'attack.tsv'):
        if task in ['FC']:
            text = line.split('\t')[2].strip().replace('\\n', '\n') + '~' + line.split('\t')[3].strip().replace('\\n',
                                                                                                                '\n')
        else:
            text = line.split('\t')[2].strip().replace('\\n', '\n')
        text_lowernospaces = text.lower().replace(' ', '')
        newlines_here = set()
        for i in range(len(text_lowernospaces)):
            if text_lowernospaces[i] == '\n':
                prefix = text_lowernospaces[(i - 10):i].replace('\n', '')[-5:]
                suffix = text_lowernospaces[i:(i + 10)].replace('\n', '')[:5]
                key = prefix + '___' + suffix
                newlines_here.add(key)
        newlines[task][text_lowernospaces.replace('\n', '')[:200]] = newlines_here

# Creating questions
questions = {task: [] for task in tasks}
for task in tasks:
    for victim in victims:
        for original, aes in all_aes[task + '_' + victim].items():
            if len(aes) < 2:
                continue
            # Recovering newlines
            original_marked = all_originalspans[task + '_' + victim][original]['combined']
            texts = [original, original_marked, aes['TREPAT'], aes['baseline']]
            for j in range(len(texts)):
                text_old = texts[j]
                text_new = text_old
                for i in range(len(text_new)):
                    # if original.replace(' ', '') in newlines[task]:
                    if len(newlines[task][original.replace(' ', '')[:200]]) > 0:
                        if text_old[i] == ' ':
                            prefix = text_old[(i - 10):i].replace(' ', '')[-5:]
                            suffix = text_old[i:(i + 10)].replace(' ', '')[:5]
                            key = prefix + '___' + suffix
                            if key in newlines[task][original.replace(' ', '')[:200]]:
                                prefix2 = text_new[:i]
                                suffix2 = text_new[(i + 1):]
                                text_new = prefix2 + '\n' + suffix2
                texts[j] = text_new
            if random.random() > 0.5:
                text_A = texts[2]
                text_B = texts[3]
                source_A = 'TREPAT'
                source_B = scenario2baseline[task + '_' + victim]
            else:
                text_A = texts[3]
                text_B = texts[2]
                source_A = scenario2baseline[task + '_' + victim]
                source_B = 'TREPAT'
            entry = {'id': task + '.' + str(len(questions[task])), 'task': task, 'victim': victim, 'original': texts[0],
                     'original_marked': texts[1],
                     'text_A': text_A, 'source_A': source_A, 'text_B': text_B, 'source_B': source_B}
            questions[task].append(entry)
    print("Available questions in task " + task + " : " + str(len(questions[task])))

# Dividing questions
questions_for_annotators = {}
for task in tasks:
    questions_for_annotators[task] = {}
    random.shuffle(questions[task])
    common = questions[task][:int(question_no[task] * shared_part)]
    used = len(common)
    for annotator in annotators:
        questions_for_annotators[task][annotator] = questions[task][used:(used + question_no[task] - len(common))]
        used += (question_no[task] - len(common))
        questions_for_annotators[task][annotator].extend(common)
        random.shuffle(questions_for_annotators[task][annotator])

# Writing HTML
for annotator in annotators:
    with open(path_attacks / ('evaluate' + annotator + '.html'), 'w') as html_out:
        html_out.write('<!DOCTYPE html>\n<html>\n')
        html_out.write('<head><meta charset="UTF-8">\n')
        html_out.write(
            '<style>\ntable {\n  font-family: arial, sans-serif;\n  border-collapse: collapse;\n  width: 100%;\n}\n\ntd, th {\n  border: 1px solid #dddddd;\n  text-align: left;\n  padding: 8px;\n}\n\ntr:nth-child(even) {\n  background-color: #dddddd;\n}\n</style>')
        html_out.write('</head><body>\n')
        html_out.write('<h1>Text comparison ' + annotator + '</h1>\n')
        for task in tasks:
            html_out.write('<h2>' + task + '</h2>\n')
            html_out.write('<table>\n')
            html_out.write(
                '<tr><th style="width:10%">ID</th><th style="width:30%">Original text</th><th style="width:30%">Modified text A</th><th style="width:30%">Modified text B</th>\n')
            for question in questions_for_annotators[task][annotator]:
                text_original = '• ' + question['original_marked'].replace('&', '&amp;').replace('&amp;# ',
                                                                                                 '&#').replace(
                    '<',
                    '[[b style="color:SeaGreen;"]]').replace(
                    '>', '[[/b]]').replace(
                    '[[', '<').replace(']]', '>').replace('\n', '<br />• ').replace(' ~ ', '<br />→ ')
                text_a = '• ' + question['text_A'].replace('&', '&amp;').replace('&amp;# ', '&#').replace('<',
                                                                                                          '[[b style="color:Crimson;"]]').replace(
                    '>', '[[/b]]').replace(
                    '[[', '<').replace(']]', '>').replace('\n', '<br />• ').replace(' ~ ', '<br />→ ')
                text_b = '• ' + question['text_B'].replace('&', '&amp;').replace('&amp;# ', '&#').replace('<',
                                                                                                          '[[b style="color:Crimson;"]]').replace(
                    '>', '[[/b]]').replace(
                    '[[', '<').replace(']]', '>').replace('\n', '<br />• ').replace(' ~ ', '<br />→ ')
                html_out.write('<tr><td>' + question[
                    'id'] + '</td><td>' + text_original + '</td><td>' + text_a + '</td><td>' + text_b + '</td>\n')
            html_out.write('</table>\n')
        html_out.write('</body>\n</html>\n')
    
    # Writing other files
    with open(path_attacks / ('answers' + annotator + '.csv'), 'w') as answers_out:
        with open(path_attacks / ('secret' + annotator + '.csv'), 'w') as secret_out:
            answers_out.write('QUESTION,BETTER MEANING PRESERVATION (A/B),BETTER AUTHENTICITY (A/B)\n')
            secret_out.write('QUESTION,METHOD A,METHOD B,VICTIM\n')
            for task in tasks:
                for question in questions_for_annotators[task][annotator]:
                    answers_out.write(task + '.' + question['id'] + ',?,?\n')
                    secret_out.write(
                        task + '.' + question['id'] + ',' + question['source_A'] + ',' + question['source_B'] + ',' +
                        question[
                            'victim'] + '\n')
