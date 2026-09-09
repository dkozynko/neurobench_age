% Minimal local BibTeX style for an offline, deterministic manuscript build.
% The full bibliographic metadata remains in references.bib.
ENTRY
  { author
    title
    journal
    booktitle
    volume
    number
    pages
    year
    doi
    url
    eprint
  }
  {}
  {}

FUNCTION {field.or.empty}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {venue}
{ booktitle empty$
    { journal field.or.empty }
    { booktitle }
  if$
}

FUNCTION {format.entry}
{ author field.or.empty
  ". " *
  title field.or.empty *
  ". " *
  venue *
  ". " *
  year field.or.empty *
  volume empty$
    'skip$
    { ", " * volume * }
  if$
  number empty$
    'skip$
    { "(" * number * ")" * }
  if$
  pages empty$
    'skip$
    { ", " * pages * }
  if$
  doi empty$
    'skip$
    { ". doi:" * doi * }
  if$
  url empty$
    'skip$
    { ". \url{" * url * "}" * }
  if$
  eprint empty$
    'skip$
    { ". arXiv:" * eprint * }
  if$
  "." *
}

FUNCTION {article}
{ "\bibitem{" cite$ * "}" * write$ newline$
  format.entry write$ newline$
}

FUNCTION {inproceedings} { article }
FUNCTION {misc} { article }
FUNCTION {default.type} { article }

FUNCTION {begin.bib}
{ "\begin{thebibliography}{99}" write$ newline$
}

FUNCTION {end.bib}
{ "\end{thebibliography}" write$ newline$
}

READ
EXECUTE { begin.bib }
ITERATE { call.type$ }
EXECUTE { end.bib }
