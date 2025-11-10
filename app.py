import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Configuracion de la pagina
st.set_page_config(
    page_title="Prediccion de Cancelaciones",
    page_icon="🏨",
    layout="wide"
)

# Titulo principal
st.title("Sistema de Prediccion de Cancelaciones de Reservas Hoteleras")
st.markdown("---")

# Cargo los modelos entrenados
@st.cache_resource
def cargar_modelos():
    try:
        modelo_lr = joblib.load('modelo_lr.pkl')
        modelo_knn = joblib.load('modelo_knn.pkl')
        scaler = joblib.load('scaler.pkl')
        return modelo_lr, modelo_knn, scaler
    except:
        st.error("Error al cargar los modelos. Asegurate de haber ejecutado el notebook primero.")
        return None, None, None

modelo_lr, modelo_knn, scaler = cargar_modelos()

if modelo_lr is not None:
    st.success("Modelos cargados correctamente")
    
    # Seleccion del modelo
    st.sidebar.header("Configuracion")
    modelo_seleccionado = st.sidebar.selectbox(
        "Selecciona el modelo:",
        ["Regresion Logistica", "KNN"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Informacion del Modelo")
    if modelo_seleccionado == "Regresion Logistica":
        st.sidebar.info("**Regresion Logistica**\n\nAccuracy: 80.83%\nTiempo: 0.53s\nRapido y eficiente")
    else:
        st.sidebar.info("**KNN**\n\nAccuracy: 84.84%\nAUC: 0.9187\nMayor precision")
else:
    st.stop()

st.markdown("---")

# Formulario de entrada de datos
st.header("Ingresa los datos de la reserva")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Informacion General")
    hotel = st.selectbox("Tipo de Hotel", ["Resort Hotel", "City Hotel"])
    lead_time = st.number_input("Dias de anticipacion", min_value=0, max_value=737, value=50)
    arrival_month = st.selectbox("Mes de llegada", 
        ["January", "February", "March", "April", "May", "June", 
         "July", "August", "September", "October", "November", "December"])
    arrival_week = st.slider("Semana del año", 1, 53, 27)
    arrival_day = st.slider("Dia del mes", 1, 31, 15)
    
with col2:
    st.subheader("Estancia")
    weekend_nights = st.number_input("Noches de fin de semana", min_value=0, max_value=19, value=0)
    week_nights = st.number_input("Noches entre semana", min_value=0, max_value=50, value=2)
    adults = st.number_input("Adultos", min_value=1, max_value=4, value=2)
    children = st.number_input("Niños", min_value=0, max_value=3, value=0)
    babies = st.number_input("Bebes", min_value=0, max_value=2, value=0)
    meal = st.selectbox("Plan de comida", ["BB", "HB", "FB", "SC"])

with col3:
    st.subheader("Detalles de Reserva")
    country = st.selectbox("Pais", ["PRT", "GBR", "USA", "ESP", "IRL", "FRA", "DEU", "ITA", "BEL", "BRA", "Other"])
    market_segment = st.selectbox("Segmento", ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Groups", "Aviation", "Complementary"])
    distribution_channel = st.selectbox("Canal", ["Direct", "Corporate", "TA/TO", "GDS"])
    is_repeated_guest = st.selectbox("Cliente repetido", [0, 1])
    previous_cancellations = st.number_input("Cancelaciones previas", min_value=0, max_value=26, value=0)
    previous_bookings = st.number_input("Reservas previas", min_value=0, max_value=72, value=0)
    
col4, col5 = st.columns(2)

with col4:
    st.subheader("Habitaciones")
    reserved_room = st.selectbox("Habitacion reservada", ["A", "B", "C", "D", "E", "F", "G", "H", "L"])
    assigned_room = st.selectbox("Habitacion asignada", ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K"])
    booking_changes = st.number_input("Cambios en la reserva", min_value=0, max_value=21, value=0)
    
with col5:
    st.subheader("Adicionales")
    deposit_type = st.selectbox("Tipo de deposito", ["No Deposit", "Refundable", "Non Refund"])
    agent = st.number_input("ID Agente", min_value=-1, max_value=535, value=-1)
    company = st.number_input("ID Empresa", min_value=-1, max_value=543, value=-1)
    days_waiting = st.number_input("Dias en lista de espera", min_value=0, max_value=391, value=0)
    customer_type = st.selectbox("Tipo de cliente", ["Transient", "Contract", "Transient-Party", "Group"])
    adr = st.number_input("Tarifa diaria promedio", min_value=0.0, max_value=5400.0, value=100.0)
    parking = st.number_input("Espacios de estacionamiento", min_value=0, max_value=8, value=0)
    special_requests = st.number_input("Solicitudes especiales", min_value=0, max_value=5, value=0)

st.markdown("---")

# Funcion para preprocesar los datos de entrada
def preprocesar_entrada(datos):
    # Codifico variables categoricas igual que en el entrenamiento
    hotel_encoded = 1 if datos['hotel'] == "Resort Hotel" else 0
    
    # Label encoding para las demas categoricas
    meal_map = {"BB": 0, "HB": 1, "SC": 2, "FB": 3}
    market_map = {"Direct": 0, "Corporate": 1, "Online TA": 2, "Offline TA/TO": 3, "Groups": 4, "Complementary": 5, "Aviation": 6}
    channel_map = {"Direct": 0, "Corporate": 1, "TA/TO": 2, "GDS": 3}
    room_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9, "L": 10}
    deposit_map = {"No Deposit": 0, "Refundable": 1, "Non Refund": 2}
    customer_map = {"Transient": 0, "Contract": 1, "Transient-Party": 2, "Group": 3}
    
    # OneHot encoding para mes (11 columnas - excluye el primer mes)
    # month_0 = February, month_1 = March, ..., month_10 = December
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    month_encoded = [1 if datos['arrival_month'] == months[i+1] else 0 for i in range(11)]
    
    # OneHot encoding para pais (10 columnas - excluye el primer pais)
    # Los 10 paises mas frecuentes + Other, drop first
    countries_ordered = ["PRT", "GBR", "USA", "ESP", "IRL", "FRA", "DEU", "ITA", "BEL", "BRA", "Other"]
    country_encoded = [1 if datos['country'] == countries_ordered[i+1] else 0 for i in range(10)]
    
    # Creo el vector de features en el orden correcto (26 features base + 11 mes + 10 pais = 47)
    features = [
        hotel_encoded,
        datos['lead_time'],
        datos['arrival_week'],
        datos['arrival_day'],
        datos['weekend_nights'],
        datos['week_nights'],
        datos['adults'],
        datos['children'],
        datos['babies'],
        meal_map.get(datos['meal'], 0),
        market_map.get(datos['market_segment'], 0),
        channel_map.get(datos['distribution_channel'], 0),
        datos['is_repeated_guest'],
        datos['previous_cancellations'],
        datos['previous_bookings'],
        room_map.get(datos['reserved_room'], 0),
        room_map.get(datos['assigned_room'], 0),
        datos['booking_changes'],
        deposit_map.get(datos['deposit_type'], 0),
        datos['agent'],
        datos['company'],
        datos['days_waiting'],
        customer_map.get(datos['customer_type'], 0),
        datos['adr'],
        datos['parking'],
        datos['special_requests']
    ]
    
    # Agrego las columnas OneHot (11 + 10 = 21 columnas)
    features.extend(month_encoded)
    features.extend(country_encoded)
    
    return np.array(features).reshape(1, -1)

# Boton para predecir
if st.button("Predecir Cancelacion", type="primary", use_container_width=True):
    # Recopilo todos los datos
    datos_entrada = {
        'hotel': hotel,
        'lead_time': lead_time,
        'arrival_month': arrival_month,
        'arrival_week': arrival_week,
        'arrival_day': arrival_day,
        'weekend_nights': weekend_nights,
        'week_nights': week_nights,
        'adults': adults,
        'children': children,
        'babies': babies,
        'meal': meal,
        'country': country,
        'market_segment': market_segment,
        'distribution_channel': distribution_channel,
        'is_repeated_guest': is_repeated_guest,
        'previous_cancellations': previous_cancellations,
        'previous_bookings': previous_bookings,
        'reserved_room': reserved_room,
        'assigned_room': assigned_room,
        'booking_changes': booking_changes,
        'deposit_type': deposit_type,
        'agent': agent,
        'company': company,
        'days_waiting': days_waiting,
        'customer_type': customer_type,
        'adr': adr,
        'parking': parking,
        'special_requests': special_requests
    }
    
    # Preproceso los datos
    X_input = preprocesar_entrada(datos_entrada)
    
    # Normalizo
    X_scaled = scaler.transform(X_input)
    
    # Selecciono el modelo
    modelo = modelo_lr if modelo_seleccionado == "Regresion Logistica" else modelo_knn
    
    # Hago la prediccion
    prediccion = modelo.predict(X_scaled)[0]
    probabilidad = modelo.predict_proba(X_scaled)[0]
    
    # Muestro los resultados
    st.markdown("---")
    st.header("Resultado de la Prediccion")
    
    col_r1, col_r2, col_r3 = st.columns(3)
    
    with col_r1:
        if prediccion == 1:
            st.error("### 🚨 ALTA PROBABILIDAD DE CANCELACION")
        else:
            st.success("### ✅ BAJA PROBABILIDAD DE CANCELACION")
    
    with col_r2:
        st.metric("Probabilidad de NO cancelar", f"{probabilidad[0]*100:.1f}%")
        st.metric("Probabilidad de cancelar", f"{probabilidad[1]*100:.1f}%")
    
    with col_r3:
        if probabilidad[1] > 0.7:
            st.warning("**Nivel de Riesgo: ALTO**\n\nRecomendaciones:\n- Solicitar deposito\n- Confirmar reserva 48h antes\n- Ofrecer opciones flexibles")
        elif probabilidad[1] > 0.4:
            st.info("**Nivel de Riesgo: MEDIO**\n\nRecomendaciones:\n- Enviar recordatorio\n- Verificar datos de contacto")
        else:
            st.success("**Nivel de Riesgo: BAJO**\n\nReserva segura")
    
    # Grafico de barras con las probabilidades
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 4))
    categorias = ['No Cancelara', 'Cancelara']
    valores = [probabilidad[0]*100, probabilidad[1]*100]
    colores = ['green', 'red']
    
    ax.barh(categorias, valores, color=colores, alpha=0.7)
    ax.set_xlabel('Probabilidad (%)')
    ax.set_title(f'Prediccion usando {modelo_seleccionado}')
    ax.set_xlim(0, 100)
    
    for i, v in enumerate(valores):
        ax.text(v + 1, i, f'{v:.1f}%', va='center')
    
    st.pyplot(fig)

# Boton para ejemplo rapido
st.markdown("---")
if st.button("Cargar Ejemplo de Prueba"):
    st.info("Recarga la pagina para ver los valores de ejemplo precargados")
