from setuptools import setup

with open('requirements.txt', 'r') as f:
	install_reqs = [
		s for s in [
			line.split('#', 1)[0].strip(' \t\n') for line in f
		] if s != ''
	]

setup(name='etc-ml',
	version='0.1.1',
	description='etc project',
	url='https://github.com/UFMG-Database-lab/etc',
	author='Vitor Mangaravite' 'Cecilia Nascimento',
	license='MIT',
	packages=['etc'],
	zip_safe=False,
	install_requires=install_reqs)
