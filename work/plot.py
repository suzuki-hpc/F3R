import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def figures1_2(name, savename):
	# Read results as a pands DataFrame object
	df = pd.read_csv(name)
	# Create Tag by concatenating Method (e.g., GMRES) and Prec (e.g., double)
	df["Tag"] = df["Method"] + "-" + df["Prec"]
	df = df.sort_values(by='Problem', ascending=True, kind='stable')

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
	handles, labels = ax.get_legend_handles_labels()
	tags = {
		"BiCGStab-double": "fp64-BiCGStab", "BiCGStab-float": "fp32-BiCGStab",
		"BiCGStab-_Float16": "fp16-BiCGStab", "BiCGStab-__half": "fp16-BiCGStab",
		"CG-double": "fp64-CG", "CG-float": "fp32-CG",
		"CG-_Float16": "fp16-CG", "CG-__half": "fp16-CG",
		"GMRES-double": "fp64-FGMRES(64)", "GMRES-float": "fp32-FGMRES(64)",
		"GMRES-_Float16": "fp16-FGMRES(64)", "GMRES-__half": "fp16-FGMRES(64)",
		"F3R-double": "fp64-F3R", "F3R-float": "fp32-F3R",
		"F3R-_Float16": "fp16-F3R", "F3R-__half": "fp16-F3R", "F3R-Best": "fp16-F3R-best",
	}
	labels = [tags[x] for x in labels]
	ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

	ax.set_ylabel("Speedup over fp64-F3R")
	ax.set_xlabel("")
	ax.set_axisbelow(True)
	ax.grid(True, axis='y')
	ax.set_ylim(bottom=-0.05)

	plt.tight_layout()
	# plt.show()
	plt.savefig(savename)

def figure3(name, savename):
	palette = ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']
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
	sns.scatterplot(a, hue='Tag', x='RelRate', y='RelSpeed', ax=axes[0], style="Tag", palette=palette, edgecolor="black")
	sns.scatterplot(b, hue='Tag', x='RelRate', y='RelSpeed', ax=axes[1], style="Tag", palette=palette, edgecolor="black")
	sns.scatterplot(c, hue='Tag', x='RelRate', y='RelSpeed', ax=axes[2], style="Tag", palette=palette, edgecolor="black")

	labels0 = [r'$m_4 = 1$', r'$m_4 = 3$', r'$m_4 = 4$']
	labels1 = [r'$m_3 = 2$', r'$m_3 = 3$', r'$m_3 = 5$', r'$m_3 = 6$']
	labels2 = [r'$m_2 = 6$', r'$m_2 = 7$', r'$m_2 = 9$', r'$m_2 = 10$']

	handles0, _ = axes[0].get_legend_handles_labels()
	axes[0].legend(handles0, labels0)
	handles1, _ = axes[1].get_legend_handles_labels()
	axes[1].legend(handles1, labels1)
	handles2, _ = axes[2].get_legend_handles_labels()
	axes[2].legend(handles2, labels2)

	for ax in axes:
		ax.set_xlim(-0.3, 1.8)
		ax.set_ylim(-0.3, 1.8)
		ax.grid(True)
		ax.set_xlabel("Relative convergence speed")
		ax.set_ylabel("Relative performance")

	plt.tight_layout()
	plt.savefig(savename)

def figure4(name, savename):
	palette = ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']
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
	sns.scatterplot(sub, hue='Method', x='RelRate', y='RelSpeed', ax=ax, style="Method", palette=palette, edgecolor="black")

	labels = ['F2', 'fp16-F2', 'F3', 'fp16-F3', 'F4']
	handles, _ = ax.get_legend_handles_labels()
	ax.legend(handles, labels)

	ax.set_xlim(-0.1, 1.8)
	ax.set_ylim(-0.1, 1.8)
	ax.grid(True)
	ax.set_xlabel("Relative convergence speed")
	ax.set_ylabel("Relative performance")

	plt.tight_layout()
	plt.savefig(savename)

def figure5(name, savename):
	palette = ["#ffffcc","#c7e9b4","#7fcdbb","#41b6c4","#2c7fb8","#253494"]
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
	sns.scatterplot(sub, hue='W', x='RelRate', y='RelSpeed', ax=ax, style="W", palette=palette, edgecolor="black")

	labels = [r'$c=1$', r'$c=4$', r'$c=16$', r'$c=32$', r'$c=128$', r'$c=256$']
	handles, _ = ax.get_legend_handles_labels()
	ax.legend(handles, labels)

	ax.set_xlim(-0.1, 1.8)
	ax.set_ylim(-0.1, 1.8)
	ax.grid(True)
	ax.set_xlabel("Relative convergence speed")
	ax.set_ylabel("Relative performance")

	plt.tight_layout()
	plt.savefig(savename)

def figure6(name, savename):
	palette=[
    "#ffffcc","#c7e9b4","#7fcdbb","#41b6c4","#1d91c0","#225ea8","#0c2c84"
	]
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
	sns.barplot(sub, hue='W', x='Problem', y='RelSpeed', ax=axes[0], palette=palette, edgecolor="black")
	sns.barplot(sub, hue='W', x='Problem', y='RelRate', ax=axes[1], palette=palette, edgecolor="black")

	for ax in axes:
		ax.set_axisbelow(True)
		ax.grid(True)
		ax.legend(ncol=7)
		ax.set_xlabel("")
	
	axes[0].set_ylabel("Performance")
	axes[1].set_ylabel("Convergence speed")

	plt.tight_layout()
	plt.savefig(savename)

def table(name1, name2):
	# Read results as a pands DataFrame object
	df1 = pd.read_csv(name1)
	df2 = pd.read_csv(name2)

	df = pd.concat([df1, df2])
	df["Tag"] = df["Method"] + "-" + df["Prec"]
	df['Iter'] = df.apply(lambda row: np.nan if row['ImplRes'] > 1.e-8 else row["Iter"], axis = 1)

	cg = df.Tag.eq("CG-double")
	bi = df.Tag.eq("BiCGStab-double")
	gm = df.Tag.eq("GMRES-double")
	f3r = df.Tag.eq("F3R-double") | df.Tag.eq("F3R-float") | df.Tag.eq("F3R-_Float16")

	col = ["Problem", "Tag", "Iter"]

	df = df[cg | bi | gm | f3r][col]
	df["Iter"] = df["Iter"].astype('Int64')
	table = df.pivot(index='Problem', columns='Tag', values='Iter')
	ta = table[['CG-double', 'BiCGStab-double', 'GMRES-double', 'F3R-double', 'F3R-float', 'F3R-_Float16']].reset_index()
	ta.to_string('table.txt', index=False)

import sys

if __name__ == '__main__':
	if sys.argv[1] == "1":
		figures1_2("t3c-figure1a.csv", "figure1a.pdf")
		figures1_2("t3c-figure1b.csv", "figure1b.pdf")

	if sys.argv[1] == "2":
		figures1_2("t3g-figure2a.csv", "figure2a.pdf")
		figures1_2("t3g-figure2b.csv", "figure2b.pdf")

	if sys.argv[1] == "3":
		figure3("t3c-figure3.csv", "figure3.pdf")

	if sys.argv[1] == "4":
		figure4("t3c-figure4.csv", "figure4.pdf")

	if sys.argv[1] == "5":
		figure5("t3c-figure5.csv", "figure5.pdf")

	if sys.argv[1] == "6":
		figure6("t3c-figure6.csv", "figure6.pdf")

	if sys.argv[1] == "table":
		table("t3c-figure1a.csv", "t3c-figure1b.csv")


