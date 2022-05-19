library(ggplot2)
library(ggrepel)

data = read.csv("xa_50_epoch_df.csv")

ggplot(data = data, aes(x = Dimension.1, y = Dimension.2, colour=Continent) +
  geom_point()
  
  # geom_text_repel(aes(label = z), 
  #                 box.padding = unit(0.45, "lines"),max.overlaps=100) +
  
  # geom_point(colour = c(rgb(174,0,1,maxColorValue=255),"green")[res$cluster], size = 3) +
  # labs(x=expression(paste(hat(bold(X)), " (first dimension)")), y=expression(paste(hat(bold(X)), " (second dimension)"))) +
  # theme(axis.title.x= element_text(size=20),
  #       axis.text.x=element_text(size=15),
  #       axis.title.y=element_text(size=20),
  #       axis.text.y=element_text(size=15),
  # )