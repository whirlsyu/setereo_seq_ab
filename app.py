from flask import Flask, render_template, send_from_directory, jsonify
from collections import OrderedDict
import os
import time

app = Flask(__name__)

def get_png_files():
    png_files = [f for f in os.listdir('results') if f.endswith('.png')]
    return png_files

def group_by_sample(png_files):
    samples = {}
    for file in png_files:
        if file.startswith('merged'):
            sample = 'merged'
        else:
            sample = file.split('_')[0]
        if sample not in samples:
            samples[sample] = []
        samples[sample].append(file)
    
    sorted_samples = OrderedDict(sorted(samples.items(), key=lambda x: custom_sort_key(x[0])))
    return sorted_samples

def custom_sort_key(sample):
    if sample == 'merged':
        return (float('inf'), float('inf'))
    parts = sample.split('-')
    if len(parts) >= 2 and parts[0].startswith('S') and parts[0][1:].isdigit() and parts[1].isdigit():
        return (int(parts[0][1:]), int(parts[1]))
    else:
        return (float('inf'), float('inf'))

def group_by_keyword(png_files):
    keywords = {}
    cell_types = ['Guard_cell', 'Lower_epidermal_cell', 'Palisade_mesophyll_cell', 
                  'Spongy_mesophyll_cell', 'Upper_epidermal_cell', 'Vascular_cell']
    
    for file in png_files:
        matched = False
        for keyword in ['filtered_umap_spatial_distribution','filtered_umap', 'umap', 
                        'counts_distribution', 'counts_ecdf', 'cpm_distribution', 
                        'cpm_ecdf', 'qc_scatter', 'qc_violin', 'spatial_distribution']:
            if keyword in file:
                if keyword not in keywords:
                    keywords[keyword] = []
                keywords[keyword].append(file)
                matched = True
                break
        
        if not matched:
            for cell_type in cell_types:
                if 'correlation' in file and cell_type in file:
                    key = f'correlation_{cell_type}'
                    if key not in keywords:
                        keywords[key] = []
                    keywords[key].append(file)
                    matched = True
                    break
                elif 'coexistence_probability' in file and cell_type in file:
                    key = f'coexistence_probability_{cell_type}'
                    if key not in keywords:
                        keywords[key] = []
                    keywords[key].append(file)
                    matched = True
                    break
                elif 'coexistence' in file and cell_type in file:
                    key = f'coexistence_{cell_type}'
                    if key not in keywords:
                        keywords[key] = []
                    keywords[key].append(file)
                    matched = True
                    break

    
    for keyword in keywords:
        keywords[keyword].sort()
    
    return OrderedDict(sorted(keywords.items()))

@app.route('/scroll/<category_type>/<category_name>')
def scroll_to_category(category_type, category_name):
    return render_template('index.html', scroll_to=f'{category_type}-{category_name}')

@app.route('/')
def index():
    png_files = get_png_files()
    samples = group_by_sample(png_files)
    keywords = group_by_keyword(png_files)
    return render_template('index.html', samples=samples, keywords=keywords)

@app.route('/results/<path:filename>')
def serve_image(filename):
    return send_from_directory('results', filename)

@app.route('/files')
def list_files():
    files = []
    for folder in ['.', 'STRING', 'data']:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f != '.DS_Store':  # exclude .DS_Store on macOS
                    file_path = os.path.join(folder, f)
                    if os.path.isfile(file_path):
                        file_stats = os.stat(file_path)
                        files.append({
                            'name': f,
                            'path': file_path,
                            'size': file_stats.st_size,
                            'modified': time.ctime(file_stats.st_mtime)
                        })
    files.sort(key=lambda x: x['name'].lower())
    return jsonify({'files': files})

@app.route('/download/<path:filename>')
def download_file(filename):
    directory = os.path.dirname(filename)
    file = os.path.basename(filename)
    return send_from_directory(directory, file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)