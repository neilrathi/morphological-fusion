rm(list=ls())
setwd("~/projects/morphological-fusion/")
library(tidyverse)

permutation_test = function(x, y, num_iter=10000, method="spearman") {
  standard = cor.test(x, y, method=method)[4]$estimate
  results = c()
  for(i in 1:num_iter) {
    y_shuffled = sample(y)
    result = cor.test(x, y_shuffled, method=method)[4]$estimate
    results = c(results, result)
  }
  mean(results >= standard)
}

ds = read_tsv("langdata/surp_size.csv")

# FUSION -> SMALL PARADIGM

ds %>%
  mutate(avg_surp=surp/size) %>%
  ggplot(aes(x=log(size), y=avg_surp, label=key)) +
    geom_text() +
    stat_smooth(method='lm') +
    theme_minimal()

# FUSION -> FREQUENCY

df = read_tsv("langdata/freq_surp.csv") %>%
  mutate(avgsurp=totsurp/numforms) %>%
  separate(features, into=c("feature", "lang"), sep="_")

df %>%
  ggplot(aes(x=logfreq, y=avgsurp, label=paste(feature, lang, sep="_"), color=lang, group=1)) +
  geom_hline(yintercept=0, color="black") +
  stat_smooth(method='lm') +
  geom_text(size=2) +
  theme_minimal() +
  guides(color=F) +
  labs(x="Log Frequency", y="Average Fusion")

ggsave("result_plots/feature_freq_fusion_plot.pdf", width=5, height=4)

# Without Hungarian:

df %>%
  filter(lang != "hun") %>%
  ggplot(aes(x=logfreq, y=avgsurp, label=paste(feature, lang, sep="_"), color=lang, group=1)) +
  stat_smooth(method='lm') +
  geom_text(size=2) +
  theme_minimal() +
  guides(color=F) +
  labs(x="Log Frequency", y="Average Fusion")

with(df, cor.test(logfreq, avgsurp, method="pearson"))
with(df, cor.test(logfreq, avgsurp, method="spearman"))

df %>%
  filter(lang != "hun") %>% 
  with(cor.test(logfreq, avgsurp, method="spearman")) # it's even stronger

with(df, permutation_test(logfreq, avgsurp))


