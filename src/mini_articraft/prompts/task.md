<task>
User request:

{{ prompt }}

Edit `main.py` in the run workspace and build the requested object. Meet the four
quality requirements from the system prompt. Use realistic geometry, model the
primary mechanism, support every part, and avoid unintended overlap.
<source_prompt>
`main.py` already contains a working object. The request above is a change to
that object, not a fresh build. Read the script first and understand what it
already models. Then make the smallest edit that satisfies the request and keep
the rest of the object as it is: existing parts, names, materials, and
articulations are all deliberate. Do not rebuild what was not asked about.
</source_prompt>

Start with the preloaded SDK quickstart. Before choosing a representation, use
`read` to survey the relevant SDK references and compare plausible build123d and
mesh approaches. Do not stop at the first workable API. Research enough to form
an internal geometry strategy for the major visible forms. Then implement the
object with `Part.add(shape, name=..., material=...)` for known physical surfaces
or `color=...` for plain matte ones, add prompt-specific checks, and run `compile`.

Treat every compile signal as design evidence. Preserve prompt-critical visible
geometry while you repair named defects.
<image_prompt>
After a successful compile, review the visual representation separately and
improve any major form that uses a crude substitute when a public authoring
method would fit it better.
</image_prompt>
Then return a short visible summary of the object and its main motion.
</task>
