from setuptools import setup

with open('requirements.txt', 'r') as f:
	install_reqs = [
		s for s in [
			line.split('#', 1)[0].strip(' \t\n') for line in f
		] if s != ''
	]

setup(name='netc',
	version='0.2.0',
	description='Next Environment for Text Classification project',
	url='https://github.com/UFMG-Database-lab/etc',
	author='Vitor Mangaravite' 'Cecilia Nascimento',
	license='MIT',
	packages=['netc'],
	zip_safe=False,
	install_requires=install_reqs)
