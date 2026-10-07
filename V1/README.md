# V1 (archivo histórico)

Resultados y documentación de la primera versión del pipeline, para
dibujos de líneas. Ya **no se usan** en el flujo actual (retratos, V2) y
se conservan solo como referencia.

- `README_V1.md`: README original de la V1.
- `resultados_dibujo_casa/`: salida de `--method all` sobre
  `input/dibujo_casa.png`. Las etapas comunes (01-05) aparecen una sola vez,
  y en cada subcarpeta (`contours/`, `edges/`, `skeleton/`) va lo propio de
  ese método. Los strokes en px fueron 13, 25 y 97 respectivamente.
  - El `trajectories_mm.json` de la V1 tiene las coordenadas truncadas a
    mm enteros y una escala distinta en X e Y. Ambos errores se corrigieron
    en la V2.

El código de la V1 (`preprocess.py`, `strokes.py`, `simplification.py`,
`coordinates.py`) sigue en `image_processing/` porque lo usa
`--mode drawing`.
