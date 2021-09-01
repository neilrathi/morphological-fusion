library(tidyverse)
library(rPref)
rm(list=ls())
setwd("~/Desktop/morphological-fusion/langdata/")
lang = "spa"

# GENERAL ANALYSIS
langs <- c("lat", "hun", "tur", "que", "fra", "fro", "por", "rus", "spa", "ita",
           "ara", "deu", "xcl", "hye", "klr", "ell", "ces", "pol", "fin", "mkd", "lav", "hbs")
families <- c("Romance", "Uralic", "Turkic", "Quechuan", "Romance", "Romance",
              "Romance", "Slavic", "Romance", "Romance", "Semitic", "Germanic",
              "Armenian", "Armenian", "Kiranti", "Hellenic", "Slavic", "Slavic",
              "Uralic", "Slavic", "Slavic", "Slavic")
freq_langs <- c("hun", "tur", "fra", "por", "rus", "spa", "ita", "deu", "ell",
                "ces", "pol", "fin")
langs_full <- c("Latin", "Hungarian", "Turkish", "Quechua", "French",
                "Old French", "Portuguese", "Russian", "Spanish", "Italian",
                "Arabic", "German", "Grabar", "Armenian", "Khaling", "Greek",
                "Czech", "Polish", "Finnish", "Macedonian", "Latvian", "Serbo-Croatian")
rom_langs <- c("lat", "fra", "fro", "por", "spa", "ita")
len_langs = length(langs)
len_rom = length(rom_langs)
df_list = list()
for (i in 1:len_langs) {
  lang_csv <- paste(langs[i], "_surprisals.txt", sep = "")
  df_list[[langs[i]]] <- read.csv(lang_csv, sep="\t", header=FALSE)
  df_list[[langs[i]]]$pos <- substr(df_list[[langs[i]]]$features, 1, 1)
  df_list[[langs[i]]]$lang <- langs_full[i]
  df_list[[langs[i]]]$code <- langs[i]
  df_list[[langs[i]]]$family <- families[i]
  if (langs[i] == 'fro') {
    df_list[[langs[i]]] <- df_list[[langs[i]]][df_list[[langs[i]]]$pos == 'V', ]
  }
  if (langs[i] == 'que') {
    df_list[[langs[i]]] <- df_list[[langs[i]]][df_list[[langs[i]]]$pos == 'N', ]
  }
  if (langs[i] == 'tur') {
    df_list[[langs[i]]] <- df_list[[langs[i]]][df_list[[langs[i]]]$pos == 'N', ]
  }
  for (j in 1:nrow(df_list[[langs[i]]])) {
    if (df_list[[langs[i]]][j, 'pos'] == 'V') {
      df_list[[langs[i]]][j, 'pos'] <- 'Verbs'
    }
    else if (df_list[[langs[i]]][j, 'pos'] == 'N') {
      df_list[[langs[i]]][j, 'pos'] <- 'Nouns'
    }
    else if (df_list[[langs[i]]][j, 'pos'] == 'A') {
      df_list[[langs[i]]][j, 'pos'] <- 'Adjectives'
    }
  }
}

# create plots
langs_df <- bind_rows(df_list)
langs_df$key <- paste(langs_df$lang, langs_df$pos, sep=" ")
langs_df$code_key <- paste(langs_df$code, langs_df$pos, sep=" ")
mean_fusion <- ggplot(langs_df, aes(x=reorder(key, surprisal, FUN = median), y=surprisal, fill=family))
mean_fusion <- mean_fusion + geom_boxplot(outlier.shape = NA, alpha = 0.8) + theme(legend.position = "right") + labs(y="Mean Fusion (bits)", x = "Language") + stat_summary(fun=mean, geom="point", shape=20, size=3, color="black", fill="black") + ylim(0, 50) + labs(fill = "Language Family")