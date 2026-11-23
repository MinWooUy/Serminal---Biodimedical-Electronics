# Dung cho: latexmk main.tex   (can Perl; Overleaf cung doc file nay)
$pdf_mode = 5;                      # 5 = xelatex
$xelatex = 'xelatex -interaction=nonstopmode -halt-on-error %O %S';
$bibtex_use = 2;
$out_dir = 'build';
