import subprocess
import os
import sys

peak_extend_command = "zcat " + sys.argv[1] + ''' | sort -k1,1 -k2,2n | bedtools cluster -i - > extended.bed''' 
subprocess.call(peak_extend_command,shell=True)


command = '''awk '{count[$NF]++} END {for (id in count) print id,count[id]}' extended.bed > cluster_counts.txt'''

subprocess.call(command,shell=True)
command = '''awk -v OFS="\t" 'NR==FNR {counts[$1]=$2; next} {$NF = $NF OFS counts[$NF]; print}' cluster_counts.txt extended.bed | awk '{OFS="\t"}{print $1,$2,$3,$4,$5,$6,$7,$8}' > extended_with_counts.bed'''

subprocess.call(command,shell=True)


subprocess.call('''awk '$8 == 1' extended_with_counts.bed > singletons.bed''',shell=True)
subprocess.call('''awk '$8 > 1' extended_with_counts.bed > multiples.bed''',shell=True)

subprocess.call('''awk '$2 > 1' cluster_counts.txt | awk '{print $1}' > multiple_clusters.txt''', shell=True)

clusters = []

# Use dynamic path resolution for input files
clusters_file = os.path.join(os.getcwd(), 'multiple_clusters.txt')
with open(clusters_file, 'r') as file:
    clusters = [line.strip() for line in file]
    
multiple_out_file = open('multiples_collapsed.bed','a')

for k,cluster in enumerate(clusters):
    subprocess.call("awk '$7 ==" + str(cluster) + "' multiples.bed > temp_cluster.bed",shell=True)

    subprocess.call("sort -k5,5nr temp_cluster.bed > temp_cluster.sorted.bed",shell=True)
    
    peaks_file = open('temp_cluster.sorted.bed','r')
    peaks = peaks_file.readlines()
    
    kept_peaks = []
    iterate=True
    iterator = 0
    while len(peaks) > 0:
        peak = peaks.pop(0)
        kept_peaks.append(peak)
        temp_peak_file = open('temp_peak_file', 'w')
        temp_peak_file.write(peak)
        temp_peak_file.close()

        temp_file = open('temp_file', 'w')
        for temp_peak in peaks:
            temp_file.write(temp_peak)
        temp_file.close()
        
        intersect_command = "bedtools intersect -a temp_file -b temp_peak_file -wa > temp_overlap"
        subprocess.call(intersect_command,shell=True)
        overlapping_peaks_file = open('temp_overlap','r')
        overlapping_peaks = overlapping_peaks_file.readlines()
        overlapping_peaks_file.close()

        peaks = [p for p in peaks if p not in overlapping_peaks]

    for line in kept_peaks:
        multiple_out_file.write(line)
    
    if k % 100 == 0:
        print('on cluster ' + str(k) + ' of ' + str(len(clusters)))
        
subprocess.call('cat multiples_collapsed.bed singletons.bed > all_peaks_collapsed.bed',shell=True)