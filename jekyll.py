try:
    from urllib.parse import quote  # Py 3
except ImportError:
    from urllib2 import quote  # Py 2
import os
import sys



BLOG_DIR = '/Users/ralasun/Desktop/ralasun.github.io/'

f = None
for arg in sys.argv:
    if arg.endswith('.ipynb'):
        f = arg.split('.ipynb')[0]
        break


c = get_config()
c.NbConvertApp.export_format = 'markdown'
c.MarkdownExporter.extra_template_basedirs = ['/opt/anaconda3/share/jupyter/nbconvert/templates/markdown']
c.MarkdownExporter.extra_template_paths = ['/opt/anaconda3/share/jupyter/nbconver/templates/jekyll.tpl']
c.MarkdownExporter.template_file = 'jekyll.tpl'
#c.Exporter.file_extension = 'md'

def path2support(path):
    """Turn a file path into a URL"""
    parts = path.split(os.path.sep)
    # print('{{ site.baseurl}}/images/' + os.path.basename(path))
    return '{{ site.baseurl}}/images/' + os.path.basename(path)

c.MarkdownExporter.filters = {'path2support': path2support}

if f:
    c.NbConvertApp.output_base = f.lower().replace(' ', '-')
    c.FilesWriter.build_directory = BLOG_DIR + '/_ipynb'
