You will see tow .txt files, containing gene expression matrix and sampling time labels. 

In embryonic_data_490genes.txt is a matrix, with each row represents a gene (a feature) and each column represents a cell (a sample).  The first row is the cell names, and the first column is gene names.

The matrix in embryonic_data_no_info_genes.txt is similar to the matrix above, however, genes here contains nearly no information about embryonic days

In label.txt is a vector, with each site represents the sampling day (label) of a corresponding cell in the data.txt file.

These files can be opened using wordpad or Notepad++, but notepad will have some problems. A better strategy is to directly read the file in your program.
In python you can use: 
pd.read_csv('./Gene_File.txt', sep='\t', index_col=0)
In R you can use: 
read.csv('./Gene_File.txt', sep='\t', row.names=1)
