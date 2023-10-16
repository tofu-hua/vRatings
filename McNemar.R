#########################################################
# 1-sided McNemar Test for comparing two updating rules #
#########################################################

## create a contingency table and conduct 1-sided McNemar test

McNemar_1 <- function(dat_only, dat_both){
  dat_only = dat_only[,1:5]
  dat_both = dat_both[,4:5]
  colnames(dat_both) = c('both_winner_old', 'both_loser_old')
  dat = cbind(dat_only, dat_both)
  n11 = n12 = n21 = n22 = 0
  mean_tie = 0
  both_tie = 0
  for(i in 1:nrow(dat)){
    if(dat[i,4] > dat[i,5] & dat[i,6] > dat[i,7]){
      n11 = n11 + 1 
    }else if(dat[i,4] > dat[i,5] & dat[i,6] < dat[i,7]){
      n12 = n12 + 1    # "only" correct, "both" incorrect
    }else if(dat[i,4] < dat[i,5] & dat[i,6] > dat[i,7]){
      n21 = n21 + 1    # "only" incorrect, "both" correct
    }else if(dat[i,4] < dat[i,5] & dat[i,6] < dat[i,7]){
      n22 = n22 + 1
    }
  }
  output_dat <- matrix(c(n11, n21, n12, n22), nrow = 2,
                       dimnames = list("only" = c("Correct", "InCorrect"),
                                       "both" = c("Correct", "InCorrect")))
  z0 = (n21-n12)/sqrt(n21+n12)
  pvalue = 1-pnorm(z0)
  output = list(data = output_dat,
                z0 = z0,
                pvalue = pvalue)
  
  return(output)
}


###########
# example #
###########

#  read files under the directory OUTPUT_b80_d5  #
# (created by "python3 vElo.py atp_train.csv atp_test.csv 80 5")
dat_only = read.csv('OUTPUT_vElo_b80_d5/Test_only.txt', header = TRUE)
dat_both = read.csv('OUTPUT_vElo_b80_d5/Test_both.txt', header = TRUE)

result = McNemar_1(dat_only, dat_both)
result
# This gives the result in the second row of Table 6 in the paper.  

capture.output(result, file='mcnemar.txt')  # save the result in mcnemar.txt