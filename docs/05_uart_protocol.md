# 05 UART протокол Jetson <-> ESP32

## Физика
- UART: TX/RX/GND
- Скорость: TBD
- Формат: 8N1
- Защита: watchdog на ESP32

## Протокол
Версия: см. shared/protocol/version.md

Минимальные команды:
- SET_VELOCITY(vx, vy, wz)
- STOP()
- E_STOP() (защёлка)
- GET_STATE()
- SET_SERVO(id, value)
