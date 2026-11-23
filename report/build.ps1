# Bien dich bao cao: XeLaTeX + BibTeX, ket qua nam trong report/build/main.pdf
#   .\build.ps1           bien dich day du (xelatex, bibtex, xelatex x2)
#   .\build.ps1 -Fast     chi 1 luot xelatex (khi dang soan, khong doi muc luc/tai lieu tham khao)
#   .\build.ps1 -Clean    xoa thu muc build/
#   .\build.ps1 -Open     mo PDF sau khi bien dich
param([switch]$Fast, [switch]$Clean, [switch]$Open)

Set-Location $PSScriptRoot
if ($Clean) { Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue; Write-Host "Da xoa build/"; return }

foreach ($tool in "xelatex", "bibtex") {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Error "Khong tim thay '$tool'. Cai MiKTeX hoac TeX Live, hoac dung Overleaf (xem README.md)."; exit 1
    }
}
New-Item -ItemType Directory -Force build | Out-Null

$opts = @("-interaction=nonstopmode", "-halt-on-error", "-output-directory=build")
if (Get-Command miktex -ErrorAction SilentlyContinue) { $opts += "--enable-installer" }  # MiKTeX tu cai goi con thieu

function Run-XeLaTeX { & xelatex @opts main.tex | Out-Null; if ($LASTEXITCODE -ne 0) { Write-Host "Loi bien dich. Xem build/main.log" -ForegroundColor Red; exit 1 } }

Run-XeLaTeX
if (-not $Fast) {
    & bibtex build/main | Out-Null
    Run-XeLaTeX
    Run-XeLaTeX
}

$log = Get-Content build/main.log -Raw
$undef = ([regex]::Matches($log, "undefined")).Count
Write-Host "Xong: build/main.pdf" -ForegroundColor Green
if ($undef -gt 0) { Write-Host "Canh bao: con tham chieu/trich dan chua dinh nghia (xem build/main.log)" -ForegroundColor Yellow }
if ($Open) { Invoke-Item build/main.pdf }
