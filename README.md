# Transformer
# Proyecto Final IA: TST

Por Manuel Charaja

# Instrucciones

- Tener Visual Studio 2022 instalado
- Tener el toolkit de CUDA instalado
- Tener Python instalado con las librerias pandas y plotlibe (Opcional para la grafica)


Para instalar esta version de Transformer se necesita el Toolkit de CUDA cuda_12.6.3_windows_network.exe disponible aqui (Ademas asegurarse que la version sea compatible con la Tarjeta grafica):
https://developer.nvidia.com/cuda-12-6-3-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network

Despues tambien debe de descargar los datasets (SP500.csv y SP50020152020.csv respectivamente) incluidos en la carpeta
Estos datasets fueron obtenidos del Federal Reserve Bank of St. Louis
https://fred.stlouisfed.org/series/SP500

Una vez se instala el toolkit, En Visual Studio debe crear el proyecto de la siguiente forma:
<img width="1523" height="435" alt="image" src="https://github.com/user-attachments/assets/c0cbbdb1-35b8-4ecd-934a-c766f53d4e3c" />

En la carpeta de proyecto (junto a kernel.cu) deben estar: preprocesamiento.h, Resultado.py, SP500.csv y SP50020152020.csv

Luego, debe copiar el codigo en el archivo kernel.cu y correrlo (Asegurarse antes de incluir las librerias CUBLAS y CudART en el Linker)

La ejecucion del codigo se encuentra en este video:
https://drive.google.com/file/d/1EgTov8iKgNGGeSaCm0hewhHK5GQE2j4S/view?usp=drive_link

# Ejecucion

Para ejecutar el Transformer, una vez habiendo seguido los pasos mencionados en la seccion de Instrucciones, se debe de compilar el codigo, en caso se use Visual Studio 2022 debe darle click a este boton para ejecutarlo en Release
<img width="403" height="143" alt="image" src="https://github.com/user-attachments/assets/72dffefd-fea9-4f55-b04b-5fd27c72340d" />

Una vez Ejecutado, tomara unos segundos y le deberia aparecer lo siguiente

<img width="361" height="783" alt="image" src="https://github.com/user-attachments/assets/a4b462a8-7c74-4bbd-b4fb-fe6cdb0e503b" />

Dependiendo de la cantidad de epocas seleccionada puede demorarse un poco

Una vez termina el proceso, y si tiene las librerias de pandas y plotlib instaladas deberia obtener una vista similar a la siguiente:

Ejemplo con 1200 epocas
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/a16db355-6615-45bd-a47f-6f6a28851ed0" />
