import sys

victims = ['BiLSTM', 'BERT', 'GEMMA', 'GEMMA7B']
task = "PR2"
targeted_variant = 'False'
attacker = 'TREPAT-OLMO7B'
variants = ["REPHRASE", "PARAPHRASE", "SIMPLIFY", "FORMAL", "INFORMAL", "CHANGE"]
path_attacks = sys.argv[1]

# Gather AEs
all_aes = {}
all_originals = set()
for victim in victims:
    all_aes[task + '_' + victim] = {}
    for variant in variants:
        for line in open(
                path_attacks / (
                        'raw_dev_' + task + '_' + targeted_variant + '_TREPAT-OLMO7B-' + variant + '_' + victim + '.tsv')):
            parts = line.strip().split('\t')
            original = parts[1].replace('>', '').replace('<', '')
            ae = parts[2]
            if original not in all_originals:
                all_originals.add(original)
            if original not in all_aes[task + '_' + victim]:
                all_aes[task + '_' + victim][original] = {}
            all_aes[task + '_' + victim][original][variant] = ae

# Writing HTML
with open(path_attacks / ('prompts.html'), 'w') as html_out:
    html_out.write('<!DOCTYPE html>\n<html>\n')
    html_out.write('<head><meta charset="UTF-8">\n')
    html_out.write(
        '<style>\ntable {\n  font-family: arial, sans-serif;\n  border-collapse: collapse;\n  width: 100%;\n}\n\ntd, th {\n  border: 1px solid #dddddd;\n  text-align: left;\n  padding: 8px;\n}\n\ntr:nth-child(even) {\n  background-color: #dddddd;\n}\n</style>')
    html_out.write('</head><body>\n')
    html_out.write('<h1>Prompt comparison PR2</h1>\n')
    html_out.write('<table>\n')
    html_out.write('<tr><th style="width:5%">Victim</th><th style="width:15%">Original text</th>' + ''.join(
        ['<th style="width:15%">' + variant + '</th>' for variant in variants]) + '</tr>\n')
    for original in all_originals:
        for victim in victims:
            text_original = original.replace('&', '&amp;').replace('&amp;# ', '&#').replace('\n', '<br />• ')
            html_out.write('<tr><td>' + victim + '</td><td>' + text_original + '</td>')
            for variant in variants:
                text_ae = (all_aes[task + '_' + victim][original][variant] if (original in all_aes[
                    task + '_' + victim] and variant in all_aes[task + '_' + victim][
                                                                                   original]) else 'NONE').replace('&',
                                                                                                                   '&amp;').replace(
                    '&amp;# ', '&#').replace(
                    '<', '[[b style="color:Crimson;"]]').replace('>', '[[/b]]').replace('[[', '<').replace(']]',
                                                                                                           '>').replace(
                    '\n', '<br />• ')
                html_out.write('<td>' + text_ae + '</td>')
            html_out.write('</tr>\n')
        html_out.write('<tr><th style="width:5%">Victim</th><th style="width:15%">Original text</th>' + ''.join(
            ['<th style="width:15%">' + variant + '</th>' for variant in variants]) + '</tr>\n')
    html_out.write('</table>\n')
    html_out.write('</body>\n</html>\n')

print("DONE")
