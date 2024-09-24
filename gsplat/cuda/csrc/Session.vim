let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/build
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +1 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/projection.cu
badd +6 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/bindings.h
badd +15 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/fully_fused_projection_2dgs_fwd.cu
badd +1 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/fully_fused_projection_2dgs_bwd.cu
badd +174 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/fully_fused_projection_radegs_fwd.cu
badd +642 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
badd +356 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
badd +21 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/fully_fused_projection_fwd.cu
badd +11 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/CMakeLists.txt
badd +7 ~/.config/nvim/coc-settings.json
badd +2 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/compile_commands.json
badd +1 CMakeFiles/gsplat.dir/includes_CUDA.rsp
badd +1 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/_wrapper.py
badd +99 ../fully_fused_projection_rade_bwd.cu
badd +72 ../fully_fused_projection_bwd.cu
badd +20 ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu
badd +3 ~/vision/diff-gaussian-rasterization/cuda_rasterizer/backward.cu
badd +151 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/rasterize_to_pixels_fwd.cu
badd +138 ../rasterize_to_pixels_2dgs_fwd.cu
badd +608 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/_torch_impl.py
badd +2 compile_commands.json
badd +1559 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/rendering.py
badd +2 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/utils.py
badd +2 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/_helper.py
badd +2 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/distributed.py
badd +1 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/relocation.py
badd +2 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/_backend.py
badd +1 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/_torch_impl_2dgs.py
badd +1 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/__init__.py
badd +33 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/__init__.py
badd +57 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/fully_fused_projection_packed_2dgs_bwd.cu
badd +432 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/rasterize_to_pixels_bwd.cu
badd +265 ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
badd +31 ~/vision/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
badd +48 ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h
badd +2 ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
badd +174 ../rasterize_to_pixels_rade_fwd.cu
badd +2 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/rasterize_to_pixels_2dgs_bwd.cu
badd +9 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/rasterize_to_indices_in_range.cu
badd +2 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/rasterize_to_indices_in_range_2dgs.cu
badd +2 ~/vision/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
badd +0 ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/rasterize_points.cu
badd +1 ../../../../../diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h
badd +0 /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/tests/test_2dgs.py
argglobal
%argdel
set lines=48 columns=191
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabnew +setlocal\ bufhidden=wipe
tabrewind
edit /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/fully_fused_projection_radegs_fwd.cu
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 127 + 95) / 191)
exe 'vert 2resize ' . ((&columns * 62 + 95) / 191)
exe 'vert 3resize ' . ((&columns * 0 + 95) / 191)
argglobal
balt ~/.config/nvim/coc-settings.json
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 161 - ((7 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 161
normal! 05|
wincmd w
argglobal
if bufexists(fnamemodify("/media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu", ":p")) | buffer /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu | else | edit /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu | endif
if &buftype ==# 'terminal'
  silent file /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
endif
balt /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/fully_fused_projection_radegs_fwd.cu
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 791 - ((23 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 791
normal! 019|
wincmd w
argglobal
if bufexists(fnamemodify("/media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/diff-gaussian-rasterization/cuda_rasterizer/forward.cu", ":p")) | buffer /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/diff-gaussian-rasterization/cuda_rasterizer/forward.cu | else | edit /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/diff-gaussian-rasterization/cuda_rasterizer/forward.cu | endif
if &buftype ==# 'terminal'
  silent file /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
endif
balt /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/fully_fused_projection_radegs_fwd.cu
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 84 - ((10 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 84
normal! 014|
wincmd w
exe 'vert 1resize ' . ((&columns * 127 + 95) / 191)
exe 'vert 2resize ' . ((&columns * 62 + 95) / 191)
exe 'vert 3resize ' . ((&columns * 0 + 95) / 191)
tabnext
edit /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/tests/test_2dgs.py
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
balt /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/fully_fused_projection_radegs_fwd.cu
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 112 - ((24 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 112
normal! 023|
tabnext
edit ../fully_fused_projection_rade_bwd.cu
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
wincmd =
argglobal
balt /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/fully_fused_projection_2dgs_bwd.cu
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 76 - ((10 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 76
normal! 037|
wincmd w
argglobal
if bufexists(fnamemodify("~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu", ":p")) | buffer ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu | else | edit ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu | endif
if &buftype ==# 'terminal'
  silent file ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu
endif
balt ../fully_fused_projection_bwd.cu
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
silent! normal! zE
1,9fold
23,144fold
203,208fold
386,387fold
412,419fold
438,452fold
494,563fold
584,588fold
691,691fold
795,803fold
835,838fold
1020,1025fold
1055,1056fold
1079,1079fold
let &fdl = &fdl
203
normal! zo
386
normal! zo
412
normal! zo
let s:l = 701 - ((21 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 701
normal! 036|
wincmd w
argglobal
if bufexists(fnamemodify("~/vision/diff-gaussian-rasterization/cuda_rasterizer/backward.cu", ":p")) | buffer ~/vision/diff-gaussian-rasterization/cuda_rasterizer/backward.cu | else | edit ~/vision/diff-gaussian-rasterization/cuda_rasterizer/backward.cu | endif
if &buftype ==# 'terminal'
  silent file ~/vision/diff-gaussian-rasterization/cuda_rasterizer/backward.cu
endif
balt ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
silent! normal! zE
let &fdl = &fdl
let s:l = 178 - ((4 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 178
normal! 0
wincmd w
wincmd =
tabnext
edit ../rasterize_to_pixels_rade_fwd.cu
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
wincmd =
argglobal
balt ../rasterize_to_pixels_2dgs_fwd.cu
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 195 - ((21 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 195
normal! 034|
wincmd w
argglobal
if bufexists(fnamemodify("~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu", ":p")) | buffer ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu | else | edit ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu | endif
if &buftype ==# 'terminal'
  silent file ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
endif
balt /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/rasterize_to_pixels_fwd.cu
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
1
normal! zo
240
normal! zo
293
normal! zo
351
normal! zo
480
normal! zo
let s:l = 371 - ((11 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 371
normal! 027|
wincmd w
argglobal
if bufexists(fnamemodify("~/vision/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu", ":p")) | buffer ~/vision/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu | else | edit ~/vision/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu | endif
if &buftype ==# 'terminal'
  silent file ~/vision/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
endif
balt ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
1
normal! zo
184
normal! zo
227
normal! zo
278
normal! zo
375
normal! zo
let s:l = 298 - ((11 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 298
normal! 01|
wincmd w
wincmd =
tabnext
edit ../rasterize_to_pixels_rade_fwd.cu
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
wincmd =
argglobal
balt /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 204 - ((30 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 204
normal! 032|
wincmd w
argglobal
if bufexists(fnamemodify("/media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu", ":p")) | buffer /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu | else | edit /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu | endif
if &buftype ==# 'terminal'
  silent file /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
endif
balt ~/vision/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
setlocal fdm=diff
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 349 - ((11 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 349
normal! 07|
wincmd w
wincmd =
tabnext
edit ~/vision/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
argglobal
balt /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
silent! normal! zE
let &fdl = &fdl
let s:l = 65 - ((22 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 65
normal! 024|
tabnext
edit ../../../../../diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h
argglobal
balt ~/vision/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 140 - ((14 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 140
normal! 027|
tabnext
edit ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
wincmd =
argglobal
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 301 - ((25 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 301
normal! 050|
wincmd w
argglobal
if bufexists(fnamemodify("~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/rasterize_points.cu", ":p")) | buffer ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/rasterize_points.cu | else | edit ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/rasterize_points.cu | endif
if &buftype ==# 'terminal'
  silent file ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/rasterize_points.cu
endif
balt ~/vision/RaDe-GS/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 77 - ((1 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 77
normal! 018|
wincmd w
wincmd =
tabnext
edit /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/_torch_impl_2dgs.py
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
1wincmd h
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
wincmd =
argglobal
balt /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/_torch_impl.py
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 102 - ((11 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 102
normal! 0101|
wincmd w
argglobal
if bufexists(fnamemodify("/media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/rendering.py", ":p")) | buffer /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/rendering.py | else | edit /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/rendering.py | endif
if &buftype ==# 'terminal'
  silent file /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/rendering.py
endif
balt /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/rasterize_to_pixels_fwd.cu
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 1558 - ((23 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 1558
normal! 0
wincmd w
wincmd =
tabnext
edit /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/rasterize_to_pixels_bwd.cu
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
balt /media/paul/f089d57e-8515-45df-bdf1-f1a9bf7077c9/vision/gsplat/gsplat/cuda/csrc/fully_fused_projection_packed_2dgs_bwd.cu
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 432 - ((18 * winheight(0) + 22) / 45)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 432
normal! 013|
tabnext 4
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
let g:this_session = v:this_session
let g:this_obsession = v:this_session
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
