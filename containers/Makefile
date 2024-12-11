gapit.sif: multtest_2.8.0.tar.gz GAPIT_3.4.0.tar.gz
	sudo singularity build gapit.sif gapit.def
	
multtest_2.8.0.tar.gz: 
	wget https://cran.r-project.org/src/contrib/Archive/multtest/multtest_2.8.0.tar.gz

GAPIT_3.4.0.tar.gz: 
	git clone https://github.com/jiabowang/GAPIT
	R CMD build GAPIT
	rm -r ./GAPIT/
