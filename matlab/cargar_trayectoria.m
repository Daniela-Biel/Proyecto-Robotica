%% cargar_trayectoria.m
% Lee la trayectoria generada por el pipeline de Python (robot_waypoints.csv),
% la grafica, verifica que todos los puntos sean alcanzables y muestra como
% recorrerla punto a punto con cinematica inversa.
%
% Formato del CSV (una fila por waypoint, en orden de ejecucion):
%   stroke_id, x_mm, y_mm, pen
%   pen = 0 -> ir a (x, y) con el lapiz ARRIBA (inicio de cada stroke)
%   pen = 1 -> ir a (x, y) con el lapiz ABAJO (dibujando)
%
% Las coordenadas estan en mm, en el marco del area de dibujo definido en
% config.py (DRAWING_ORIGIN_MM, FLIP_Y). La IK de abajo es un EJEMPLO para
% un brazo planar de 2 eslabones (2R/SCARA): reemplazala por la de tu robot.

clear; clc;

archivo = fullfile('..', 'output', 'robot_waypoints.csv');
T = readmatrix(archivo);            % columnas: stroke_id, x, y, pen
stroke = T(:, 1);
x = T(:, 2);
y = T(:, 3);
pen = T(:, 4);
fprintf('%d waypoints, %d strokes\n', numel(x), max(stroke));

%% ---------------------------------------------------------------------
%  Parametros del robot (EJEMPLO: brazo 2R) -- AJUSTAR
% ----------------------------------------------------------------------
L1 = 150;               % mm, eslabon 1
L2 = 150;               % mm, eslabon 2
base = [100, -60];      % mm, posicion de la base en el marco del papel
codo_arriba = true;     % eleccion de la solucion de IK

%% ---------------------------------------------------------------------
%  Vista previa
% ----------------------------------------------------------------------
figure('Name', 'Trayectoria'); hold on; axis equal; grid on;
for s = 1:max(stroke)
    k = stroke == s;
    plot(x(k), y(k), 'k-');
end
plot(x(pen == 0), y(pen == 0), 'r.', 'MarkerSize', 6);   % inicios de stroke
plot(base(1), base(2), 'bs', 'MarkerFaceColor', 'b');
xlabel('x [mm]'); ylabel('y [mm]');
title('Trazos (negro), inicios con lapiz arriba (rojo), base (azul)');

%% ---------------------------------------------------------------------
%  Verificacion: alcanzabilidad y tamano de segmentos
% ----------------------------------------------------------------------
r = hypot(x - base(1), y - base(2));
fuera = r > (L1 + L2) | r < abs(L1 - L2);
if any(fuera)
    warning('%d puntos fuera del espacio de trabajo. Mueve la base o reduce el area en config.py.', sum(fuera));
end
seg = hypot(diff(x), diff(y));
seg = seg(pen(2:end) == 1);          % solo segmentos dibujando
fprintf('Segmento maximo dibujando: %.2f mm\n', max(seg));

%% ---------------------------------------------------------------------
%  Recorrido punto a punto
% ----------------------------------------------------------------------
q = zeros(numel(x), 2);
for i = 1:numel(x)
    q(i, :) = ik_2r(x(i) - base(1), y(i) - base(2), L1, L2, codo_arriba);
    % --- Aqui va el envio al robot (Raspberry Pi / drivers) ---
    % if pen(i) == 0, subir_lapiz(); mover_a(q(i,:)); bajar_lapiz();
    % else,           mover_a(q(i,:)); end
end
fprintf('IK resuelta para %d waypoints.\n', size(q, 1));

figure('Name', 'Angulos articulares');
plot(rad2deg(q)); legend('q1', 'q2'); xlabel('waypoint'); ylabel('grados'); grid on;

%% ---------------------------------------------------------------------
function q = ik_2r(px, py, L1, L2, codo_arriba)
% Cinematica inversa analitica de un brazo planar de 2 eslabones.
    c2 = (px^2 + py^2 - L1^2 - L2^2) / (2 * L1 * L2);
    c2 = min(max(c2, -1), 1);                % proteger contra ruido numerico
    s2 = sqrt(1 - c2^2);
    if codo_arriba, s2 = -s2; end
    q2 = atan2(s2, c2);
    q1 = atan2(py, px) - atan2(L2 * s2, L1 + L2 * c2);
    q = [q1, q2];
end
