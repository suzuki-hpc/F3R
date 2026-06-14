#!/bin/zsh

names=(Bump_2911 Emilia_923 G3_circuit Queen_4147 Serena apache2 audikw_1 ecology2 ldoor thermal2 tmt_sym Freescale1 Transport atmosmodd atmosmodj atmosmodl rajat31 ss stokes t2em tmt_unsym vas_stokes_1M vas_stokes_2M)
groups=(Janna Janna AMD Janna Janna GHS_psdef GHS_psdef McRae GHS_psdef Schmid CEMW Freescale Janna Bourchtein Bourchtein Bourchtein Rajat VLSI VLSI CEMW CEMW VLSI VLSI)

for (( i = 1; i <= ${#names[@]}; i++ )); do
    curl -o $names[i].tar.gz http://sparse-files.engr.tamu.edu/MM/$groups[i]/$names[i].tar.gz
    tar -zxvf $names[i].tar.gz
    rm $names[i].tar.gz
    mv $names[i]/$names[i].mtx ./
    rm -rf $names[i]
done
