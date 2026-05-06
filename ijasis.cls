%=====================================================================
% ijasis.cls  --  Document class for the International Journal of
%                 Advances in Signal and Image Sciences (IJASIS)
%                 Published by XLESCIENCE
%
% Usage:  \documentclass{ijasis}
%
% Version 1.0  (2026)
%=====================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ijasis}[2026/05/01 IJASIS Journal Class v1.0]

%---------------------------------------------------------------------
% Base class
%---------------------------------------------------------------------
\LoadClass[12pt, a4paper]{article}

%---------------------------------------------------------------------
% Required packages
%---------------------------------------------------------------------
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{microtype}           % microtypography — kills overfull hboxes
\RequirePackage[margin=1in]{geometry}
\RequirePackage{amsmath, amssymb, amsthm}
\RequirePackage{graphicx}
\RequirePackage[hidelinks,
                colorlinks=true,
                linkcolor=blue!60!black,
                citecolor=blue!60!black,
                urlcolor=blue!60!black]{hyperref}
\RequirePackage{cite}
\RequirePackage{enumerate}
\RequirePackage{mathrsfs}
\RequirePackage{bm}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{multirow}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{float}
\RequirePackage{xcolor}
\RequirePackage{setspace}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{abstract}
\RequirePackage{tikz}
\RequirePackage{tikzpagenodes}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, fit, backgrounds}

%---------------------------------------------------------------------
% Page style
%---------------------------------------------------------------------
\pagestyle{fancy}
\setlength{\headheight}{14pt}
\fancyhf{}
\fancyhead[L]{\small\itshape IJASIS -- International Journal of
              Advances in Signal and Image Sciences}
\fancyhead[R]{\small\thepage}
\fancyfoot[C]{\small e-ISSN: 2457-0370 $|$ XLESCIENCE}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

%---------------------------------------------------------------------
% Section formatting
%---------------------------------------------------------------------
\titleformat{\section}{\large\bfseries}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection.}{0.5em}{}
\titleformat{\subsubsection}{\normalsize\itshape}{\thesubsubsection.}{0.5em}{}

%---------------------------------------------------------------------
% Paragraph spacing
%---------------------------------------------------------------------
\setlength{\parskip}{4pt}
\setlength{\parindent}{1.2em}
\onehalfspacing

%---------------------------------------------------------------------
% Theorem environments
%---------------------------------------------------------------------
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{definition}{Definition}[section]
\newtheorem{remark}{Remark}[section]
\newtheorem{example}{Example}[section]

%---------------------------------------------------------------------
% Custom math commands
%---------------------------------------------------------------------
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\norm}[1]{\left\|#1\right\|}
\newcommand{\abs}[1]{\left|#1\right|}
\newcommand{\inner}[2]{\langle #1,\, #2 \rangle}
\newcommand{\ds}{\displaystyle}
\newcommand{\Df}{\mathscr{D}}
\newcommand{\Lf}{\mathscr{L}}
\newcommand{\deriv}[2]{\frac{d#1}{d#2}}
\newcommand{\pderiv}[2]{\frac{\partial #1}{\partial #2}}

%---------------------------------------------------------------------
% Metadata commands (filled in by author)
%---------------------------------------------------------------------
\newcommand{\journalname}{International Journal of Advances in Signal
             and Image Sciences}
\newcommand{\journalabbrev}{IJASIS}
\newcommand{\journalissn}{e-ISSN: 2457-0370}
\newcommand{\journalpublisher}{XLESCIENCE}

%=====================================================================
\endinput
%=====================================================================
