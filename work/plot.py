import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def figures1_2(name, savename):
	# Read results as a pands DataFrame object
	df = pd.read_csv(name)
	# Create Tag by concatenating Method (e.g., GMRES) and Prec (e.g., double)
	df["Tag"] = df["Method"] + "-" + df["Prec"]

	# Compute speedups over the base line fp64-F3R
	base = df[df.Tag.eq("F3R-double")].set_index('Problem')['Time'].copy()
	plot = df[~df.Tag.eq("F3R-double")].copy()
	plot['Speedup'] = plot.apply(lambda row: base[row['Problem']] / row['Time'], axis = 1)
	# Set the result to NaN if the solution was diverged or stalled
	plot['Speedup'] = plot.apply(lambda row: np.nan if np.isnan(row['ImplRes']) else row['Speedup'], axis = 1)
	plot['Speedup'] = plot.apply(lambda row: np.nan if row['ImplRes'] > 1.e-8 else row['Speedup'], axis = 1)

	# 
	fig, ax = plt.subplots(figsize=(12, 5))

	palette = [
		'#ece7f2','#a6bddb','#2b8cbe',
		'#e5f5e0','#a1d99b','#31a354',
		'#fee8c8','#fdbb84','#e34a33']

	sns.set_theme(style="ticks")
	sns.barplot(plot, hue='Tag', x='Problem', y='Speedup', palette=palette, edgecolor='black')

	ax.set_xticklabels(ax.get_xticklabels(), rotation=40)
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	ax.set_axisbelow(True)
	ax.grid(True, axis='y')
	ax.set_ylim(bottom=-0.05)

	plt.tight_layout()
	# plt.show()
	plt.savefig(savename)

def figure3(name, savename):
	# Read results as a pands DataFrame object
	df = pd.read_csv(name)
	df["Tag"] = df["M2"].astype(str) + "-" + df["M3"].astype(str) + "-" + df["M4"].astype(str)

	baseT = df[df.Tag.eq("8-4-2")].set_index('Problem')['Time'].copy()
	baseI = df[df.Tag.eq("8-4-2")].set_index('Problem')['Iter'].copy()
	sub = df[~df.Tag.eq("8-4-2")].copy()

	sub['RelSpeed'] = sub.apply(lambda row: baseT[row['Problem']] / row['Time'], axis = 1)
	sub['RelRate'] = sub.apply(lambda row: baseI[row['Problem']] / row['Iter'], axis = 1)

	a = sub[sub.Tag.str.match(r"^8-4-\d+$")].copy()
	b = sub[sub.Tag.str.match(r"^8-\d+-2$")].copy()
	c = sub[sub.Tag.str.match(r"^\d+-4-2$")].copy()	

	fig, axes = plt.subplots(1, 3, figsize=(12, 5))

	sns.set_theme(style="ticks")
	sns.scatterplot(a, hue='Tag', x='RelRate', y='RelSpeed', ax=axes[0])
	sns.scatterplot(b, hue='Tag', x='RelRate', y='RelSpeed', ax=axes[1])
	sns.scatterplot(c, hue='Tag', x='RelRate', y='RelSpeed', ax=axes[2])

	for ax in axes:
		ax.set_xlim(-0.3, 1.8)
		ax.set_ylim(-0.3, 1.8)
		ax.grid(True)
		ax.set_xlabel("Relative convergence speed")
		ax.set_ylabel("Relative performance")

	plt.tight_layout()
	plt.savefig(savename)

def figure4(name, savename):
	# Read results as a pands DataFrame object
	df = pd.read_csv(name)

	baseT = df[df.Method.eq("F3R")].set_index('Problem')['Time'].copy()
	baseI = df[df.Method.eq("F3R")].set_index('Problem')['Iter'].copy()
	sub = df[~df.Method.eq("F3R")].copy()

	sub['RelSpeed'] = sub.apply(lambda row: baseT[row['Problem']] / row['Time'], axis = 1)
	sub['RelRate'] = sub.apply(lambda row: baseI[row['Problem']] / row['Iter'], axis = 1)

	sub['RelSpeed'] = sub.apply(lambda row: 0 if row['ImplRes'] > 1.e-8 else row['RelSpeed'], axis = 1)
	sub['RelRate'] = sub.apply(lambda row: 0 if row['ImplRes'] > 1.e-8 else row['RelRate'], axis = 1)

	fig, ax = plt.subplots(1, 1, figsize=(5, 5))

	sns.set_theme(style="ticks")
	sns.scatterplot(sub, hue='Method', x='RelRate', y='RelSpeed', ax=ax)

	ax.set_xlim(-0.1, 1.8)
	ax.set_ylim(-0.1, 1.8)
	ax.grid(True)
	ax.set_xlabel("Relative convergence speed")
	ax.set_ylabel("Relative performance")

	plt.tight_layout()
	plt.savefig(savename)

def figure5(name, savename):
	# Read results as a pands DataFrame object
	df = pd.read_csv(name)
	df["W"] = df["W"].astype(str)
	baseT = df[df.W.eq("64")].set_index('Problem')['Time'].copy()
	baseI = df[df.W.eq("64")].set_index('Problem')['Iter'].copy()
	sub = df[~df.W.eq("64")].copy()

	sub['RelSpeed'] = sub.apply(lambda row: baseT[row['Problem']] / row['Time'], axis = 1)
	sub['RelRate'] = sub.apply(lambda row: baseI[row['Problem']] / row['Iter'], axis = 1)
	sub['RelSpeed'] = sub.apply(lambda row: 0 if row['ImplRes'] > 1.e-8 else row['RelSpeed'], axis = 1)
	sub['RelRate'] = sub.apply(lambda row: 0 if row['ImplRes'] > 1.e-8 else row['RelRate'], axis = 1)

	fig, ax = plt.subplots(1, 1, figsize=(5, 5))

	sns.set_theme(style="ticks")
	sns.scatterplot(sub, hue='W', x='RelRate', y='RelSpeed', ax=ax)

	ax.set_xlim(-0.1, 1.8)
	ax.set_ylim(-0.1, 1.8)
	ax.grid(True)
	ax.set_xlabel("Relative convergence speed")
	ax.set_ylabel("Relative performance")

	plt.tight_layout()
	plt.savefig(savename)

def figure6(name, savename):
	# Read results as a pands DataFrame object
	df = pd.read_csv(name)
	df["W"] = df["W"].astype(str)
	baseT = df[df.Method.eq("F3R")].set_index('Problem')['Time'].copy()
	baseI = df[df.Method.eq("F3R")].set_index('Problem')['Iter'].copy()
	sub = df[~df.Method.eq("F3R")].copy()

	sub['RelSpeed'] = sub.apply(lambda row: baseT[row['Problem']] / row['Time'], axis = 1)
	sub['RelRate'] = sub.apply(lambda row: baseI[row['Problem']] / row['Iter'], axis = 1)
	sub['RelSpeed'] = sub.apply(lambda row: 0 if row['ImplRes'] > 1.e-8 else row['RelSpeed'], axis = 1)
	sub['RelRate'] = sub.apply(lambda row: 0 if row['ImplRes'] > 1.e-8 else row['RelRate'], axis = 1)

	fig, axes = plt.subplots(2, 1, figsize=(12, 5))

	sns.set_theme(style='ticks')
	sns.barplot(sub, hue='W', x='Problem', y='RelSpeed', ax=axes[0])
	sns.barplot(sub, hue='W', x='Problem', y='RelRate', ax=axes[1])

	for ax in axes:
		ax.set_axisbelow(True)
		ax.grid(True)
		ax.legend(ncol=7)
		# ax.set_xlabel("Relative convergence speed")
		# ax.set_ylabel("Relative performance")

	plt.tight_layout()
	plt.savefig(savename)

import sys

if __name__ == '__main__':
	if sys.argv[1] == "1":
		figures1_2("t3c-symmetric.csv", "figure1a.pdf")
		figures1_2("t3c-general.csv", "figure1b.pdf")

	if sys.argv[1] == "2":
		figures1_2("t3g-symmetric.csv", "figure2a.pdf")
		figures1_2("t3g-general.csv", "figure2b.pdf")

	if sys.argv[1] == "3":
		figure3("t3cp-figure3.csv", "figure3.pdf")

	if sys.argv[1] == "4":
		figure4("t3cp-figure4.csv", "figure4.pdf")

	if sys.argv[1] == "5":
		figure5("t3cp-figure5.csv", "figure5.pdf")

	if sys.argv[1] == "6":
		figure6("t3cp-figure6.csv", "figure6.pdf")


