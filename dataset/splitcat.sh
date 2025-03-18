#!/bin/bash

# Folder containing text files
folder="openwebtext"
output_file="merged.txt"  # Temporary file to store all content
output_file1="train.txt"
output_file2="test.txt"

# Split the merged file into 1GB chunks
echo "Splitting $output_file into 1GB chunks..."
split -b 1G $output_file "${output_file%.txt}_chunk_"
echo "Splitting completed."

# Calculate the total number of chunks created
chunk_files=(${output_file%.txt}_chunk_*)
total_chunks=${#chunk_files[@]}
seventy_percent=$(($total_chunks * 70 / 100))
thirty_percent=$(($total_chunks - $seventy_percent))

echo "Total chunks created: $total_chunks"
echo "70% of chunks will go to $output_file1, which is $seventy_percent chunks."
echo "Remaining 30% of chunks will go to $output_file2, which is $thirty_percent chunks."
echo ""

# Concatenate the first 70% of chunks into train.txt
echo "Starting to concatenate 70% of chunks into $output_file1..."
for ((i=0; i<$seventy_percent; i++)); do
    cat "${chunk_files[$i]}" >> $output_file1
    if ((i % 2 == 0)); then
        echo "Processed $((i + 1)) chunks for $output_file1. Current chunk: ${chunk_files[$i]}"
    fi
done
echo "Concatenation of $output_file1 completed."

# Concatenate the remaining 30% of chunks into test.txt
echo "Starting to concatenate 30% of chunks into $output_file2..."
for ((i=$seventy_percent; i<$total_chunks; i++)); do
    cat "${chunk_files[$i]}" >> $output_file2
    if ((i % 2 == 0)); then
        echo "Processed $((i + 1 - seventy_percent)) chunks for $output_file2. Current chunk: ${chunk_files[$i]}"
    fi
done
echo "Concatenation of $output_file2 completed."

echo "Process completed at $(date)."
