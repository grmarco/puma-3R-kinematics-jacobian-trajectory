
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
import numpy as np
import math as m

class RobotPUMA():
	
	#El constructor recibe tres vectores correspondientes con las columnas de la tabla de DH
	def __init__(self, d, a, alfa, theta):
	
		if len(d) != len(a) != len(alfa) != len(theta):	
			print("Parametros DH no estan bien establecidos")
			exit()
			
		self.d = d;
		self.a = a;
		self.alfa = alfa;
		self.theta = theta;
	
	#############################################	
	###			Cinematica directa		      ###
	#############################################
	
	def cinematica_directa(self, theta):
		self.theta = theta;
		
		#Matriz A de cada eje respecto al anterior
		A10 = self.matrizA(self.theta[0] , self.d[0], self.a[0], self.alfa[0])		
		A21 = self.matrizA(self.theta[1], self.d[1], self.a[1], self.alfa[1])
		A32 = self.matrizA(self.theta[2], self.d[2], self.a[2], self.alfa[2])
		
		#Multiplicamos respecto a la trama adjunta a la base
		A20 = np.dot(A10, A21)
		A30 = np.dot(A20, A32)

		#Devolvemos las coordenadas de cada trama adjunta a cada articulacion
		#para poder graficar el brazo entero
		return np.transpose([A10[0:3, 3], A20[0:3, 3], A30[0:3, 3]])
	
	def matrizA(self, theta_i, d_i, a_i, alfa_i):
		return np.array([[m.cos(theta_i), 	-m.sin(theta_i) * m.cos(alfa_i), 	m.sin(theta_i) * m.sin(alfa_i), 	a_i * m.cos(theta_i)], \
				[m.sin(theta_i), 			m.cos(theta_i) * m.cos(alfa_i), 	-m.sin(alfa_i)*m.cos(theta_i), 		a_i * m.sin(theta_i)], \
				[0.0, 					m.sin(alfa_i), 							m.cos(alfa_i), 						d_i], \
				[0.0, 					0.0, 									0.0, 								1.0]])
	

	#############################################	
	###			Cinematica inversa		      ###
	#############################################	
	
	def cinematica_inversa(self, posicion):
		
		theta = [0 for i in range(3)]
		px = round(posicion[0])
		py = round(posicion[1])
		pz = round(posicion[2])
		
		#Conocemos analiticamente los valores que producen una singularidad
		

		#Theta1
		if(px != 0):
			theta[0] = m.atan2(py,px)
		else:
			theta[0] = "Infinitas soluciones para theta1"	
		#Primero calculamos theta3, ya que theta2 se resuelve en funcion de esta
		cos_theta3 = (m.pow(px,2) + m.pow(py,2) + m.pow(pz,2) - m.pow(a[1],2) - m.pow(a[2],2))/(2*a[1]*a[2])
		cos_theta3_cuadrado = m.pow(cos_theta3, 2)
		
		#Posible situacion en que el coseno calculado geometricamene con la posicion del extremo
		#sea mayor que uno (no se puede alcanzar el punto por la logitud del brazo)
		#o el coseno, que se encuentra en el denominador de theta3
		#es cero
		if(abs(cos_theta3)>1 or 0==cos_theta3):
			theta[2] = "Singularidad theta3"
		else: 
			theta[2] = m.atan2(m.sqrt(1-cos_theta3_cuadrado), cos_theta3)
		
		#Finalmente calculamos theta2
				
		sen_theta3 = m.sqrt(1-m.pow(cos_theta3,2))
		
		beta = m.atan2(m.sqrt(m.pow(px,2)+m.pow(px,2)), pz)
		#gamma = m.atan2(a[1] + a[2]*m.cos(theta[2]), a[2]*m.sin(theta[2]))
		gamma = m.atan2(a[1] + a[2]*cos_theta3, a[2]*sen_theta3)
		theta[1] = gamma - beta
		
		return theta
	
	#############################################	
	###				Jacobiana				  ###
	#############################################	
	
	#Escribimos la expresion explicita alcanzada derivando
	def jacobiana(self, theta):
		return np.array([[-a[2]*m.sin(theta[0])*m.cos(theta[1]+theta[2]) -a[1]*m.sin(theta[0])*m.cos(theta[1]),	-a[1]*m.cos(theta[0])*m.sin(theta[1]) -a[2]*m.cos(theta[0])*m.sin(theta[1]+theta[2]), -a[2]*m.cos(theta[0])*m.sin(theta[1]+theta[2])],\
				[a[1]*m.cos(theta[0])*m.sin(theta[1])+ a[2]*m.cos(theta[0])*m.cos(theta[1]+theta[2]), a[1]*m.sin(theta[0])*m.cos(theta[1])  +a[2]*m.sin(theta[0])*m.sin(theta[1]+theta[2]),	-a[2]*m.sin(theta[0])*m.sin(theta[1]+theta[2])],\
				[0,	a[1]*m.cos(theta[1])+a[2]*m.cos(theta[1]+theta[2]),	a[2]*m.cos(theta[1]+theta[2])],\
				[0,	0, 0],\
				[1, 1, 1]])
	
	def jacobiana_por(self, theta, q):
		return np.dot(self.jacobiana(theta),q)
	
	#Esta funcion recibe una lista de puntos y devuelve la trayectoria que el robot
	#ha de seguir para alcanzar uno detras de otro
	def trayectoria_jacobiana_hacia(self, puntos_funcion):
		#Devolveremos la lista de las tres tramas de cada articulacion 
		#para poder dibujar el brazo
		lista_articulares = list()
		
		for i in range(len(puntos_funcion)):
		
			punto = puntos_funcion[i]
			print("Calculando punto", punto)					
			
			#Obtenemos las posiciones articulares para ese punto usando la c.directa
			posiciones_articulares = self.cinematica_directa(self.theta)
			lista_articulares.append(posiciones_articulares)
			punto_actual = [row[2] for row in posiciones_articulares]

			#Queremos alcanzar ese punto, calculamos lo cerca que estamos de el
			error = np.subtract(punto, punto_actual)
			
			#Si los puntos estan muy pegados seguimos
			if(error[0] < 0.05 and error[1] < 0.05 and error[2] < 0.05):
				continue
			
			#Establecemos el factor de velocidad
			incremento_posicion = 0.1
			qd = [0 for x in range(3)] 
			
			#Establecemos un limite de iteraciones por si se produce el caso de que un punto no es alcanzable 
			#(nunca se cumpliria la condicion del while ya que el error seria grande siempre)
			iteraciones = 0
			lim_iteraciones = 1e3
			
			while (error[0] > 0.05 or error[1] > 0.05 or error[2] > 0.05)  :
				
				if(iteraciones >= lim_iteraciones):
					print("	Punto no alcanzable ", punto)
					break
				
				#Vector de velocidad lineal (modulo, direccion y sentido de la velocidad del manipulador)
				v = ( error / len(error) ) * incremento_posicion
				
				#Obtenemos la jacobiana
				J = self.jacobiana(self.theta)
				J = J[0:3][0:3]
				
				#Si el determinante es cero calculamos la jacobiana
				if abs(np.linalg.det(J)) < 0.0005:
					#Puede ser que el determinante del producto tambien sea cero, en ese caso 
					#usamos la traspuesta para calcular la trayectoria
					JJi = np.dot(J,np.transpose(J))
					if(abs(np.linalg.det(JJi)) < 0.0005):
						Ji = np.transpose(J)
					else:
						Ji = np.dot(np.transpose(J),np.linalg.inv(JJi))						
				else:
					Ji = np.linalg.inv(J)
					
				qd = np.dot(Ji, v)

				# Incrementamos las coordenadas articulares
				self.theta[0] += qd[0];
				self.theta[1] += qd[1];
				self.theta[2] += qd[2];
				
				#Calamos el error respecto al nuevo punto
				posiciones_articulares = self.cinematica_directa(self.theta);
				punto_actual = [row[2] for row in posiciones_articulares] 
				lista_articulares.append(posiciones_articulares)
				error = np.subtract(punto, punto_actual)
				iteraciones += 1
		
			
		return lista_articulares
	
	#Funcion que recibe una funcion o composicion de funciones como string y devuelve una lista
	#de puntos de esa funcion, generados con con un paso  
	def generar_puntos_para_funcion(self, cantidad, paso, expresion, xi, yi):
		hayX, hayY = "x" in expresion, "y" in expresion
		if(not hayX and not hayY):
			print ("La expresion debe contener x o y")
			return []
			
		puntos = []
		#Usamos de limite la hipotenusa de la longitud del brazo 
		hipotenusa = m.sqrt(m.pow(self.a[1],2)+m.pow(self.a[2],2))
		x, y = xi, yi
		z = eval(expresion)

		for i in range(cantidad):
			
			if(z > hipotenusa):
				z = z % hipotenusa				
			if(hayX and hayY):
				if(i < cantidad/2):
					x += paso 
					y = y
				elif(i >= cantidad/2):
					x = x 
					y += paso
			elif(hayX):
				x += paso
			elif(hayY):
				y += paso				
			
			if(x > hipotenusa):
				x = x % (-hipotenusa)
				if(hayX and hayY):
					y += paso
			if(y > hipotenusa):
				y = y % (-hipotenusa)
				if(hayX and hayY):
					x += paso

			z = eval(expresion)
			puntos.append([x+xi, y+yi, z])
			
			
		return puntos


			
#############################################	
###				EJECUCION				  ###
#############################################	

d = [0,0,0]
a = [0, 1, 1]
alfa = [m.pi/2, 0, 0]
theta = [0, m.pi/2, 0 ]

#############################################	
###		EJEMPLOS PROPORCIONADOS  		  ###
#############################################	

theta1 = [0, 0, 0 ]
theta2 = [0, m.pi/2, 0 ]
theta3 = [-m.pi/2, m.pi/2, 0 ]
theta4 = [m.pi, 0, m.pi/2 ]

theta1_jaco = [0, 0, m.pi/2]
theta2_jaco = [0, m.pi/4, -m.pi/4]
theta3_jaco = [m.pi/2, 0, -m.pi/2]
theta4_jaco = [0, m.pi/2, 0]
q1 = [0, 0, m.pi/90]
q2 = [0, -m.pi/90, m.pi/90]
q3 = [0, m.pi/90, 0]
q4 = [0, 0, -m.pi/90]

robot_ejemplos = RobotPUMA(d, a, alfa, theta1)

#Imprimimos los ejemplos 

#EJEMPLO 1
directa1 = robot_ejemplos.cinematica_directa(theta1)
directa1_t = np.transpose(directa1)
end_effector1 = directa1_t[2]
inversa1 = robot_ejemplos.cinematica_inversa(end_effector1)

#EJEMPLO 2
directa2 = robot_ejemplos.cinematica_directa(theta2)
directa2_t = np.transpose(directa2)
end_effector2 = directa2_t[2]
inversa2 = robot_ejemplos.cinematica_inversa(end_effector2)

#EJEMPLO 3
directa3 = robot_ejemplos.cinematica_directa(theta3)
directa3_t = np.transpose(directa3)
end_effector3 = directa3_t[2]
inversa3 = robot_ejemplos.cinematica_inversa(end_effector3)

#EJEMPLO 4
directa4 = robot_ejemplos.cinematica_directa(theta4)
directa4_t = np.transpose(directa4)
end_effector4 = directa4_t[2]
inversa4 = robot_ejemplos.cinematica_inversa(end_effector4)

#Jacobianas
jacobiana1 = robot_ejemplos.jacobiana_por(theta1_jaco, q1)
jacobiana2 = robot_ejemplos.jacobiana_por(theta2_jaco, q2)
jacobiana3 = robot_ejemplos.jacobiana_por(theta3_jaco, q3)
jacobiana4 = robot_ejemplos.jacobiana_por(theta4_jaco, q4)

print("#############################################")
print("###		EJEMPLOS PROPORCIONADOS  		  ###")
print("#############################################")

print("Theta1:", theta1)
print("Theta2:", theta2)
print("Theta3:", theta3)
print("Theta4:", theta4)
print("")
print ("Directa 1:")
print (np.round(directa1, 2))
print ("Directa 2:")
print (np.round(directa2, 2))
print ("Directa 3:")
print (np.round(directa3, 2))
print ("Directa 4:")
print (np.round(directa4, 2))
print("")
print ("Inversa 1:")
print (inversa1)
print ("Inversa 2:")
print (inversa2)
print ("Inversa 3:")
print (inversa3)
print ("Inversa 4:")
print (inversa4)
print("")
print ("Jacobiana 1:")
print (np.round(jacobiana1,6))
print ("Jacobiana 2:")
print (np.round(jacobiana2,6))
print ("Jacobiana 3:")
print (np.round(jacobiana3,6))
print ("Jacobiana 4:")
print (np.round(jacobiana4,6))
print("")

#############################################	
###		GRAFICANDO EL MANIPULADOR  		  ###
#############################################	

robot = RobotPUMA(d, a, alfa, theta)
p_articulacion= robot.cinematica_directa(theta)

#Distintos tipos de funciones
#puntos_funcion = robot.generar_puntos_para_funcion(100, m.pi/2, "m.sin(y)+m.cos(x)", 0, 2)
#puntos_funcion = robot.generar_puntos_para_funcion(100, m.pi/2, "m.pow(y,2)+m.pow(x,2)", 0, 2)
puntos_funcion = robot.generar_puntos_para_funcion(100, m.pi/2, "m.cos(y)", 2, 0)
#Obtenemos la secuencia de posiciones de las coordenadas articulares que vamos a graficar
sec_coord_articulares = robot.trayectoria_jacobiana_hacia(puntos_funcion)

plt.close('all')
fig = plt.figure()

#Fila de la matriz de las posiciones x,y,z de las articulaciones de la posicion inicial del brazo
articulacion_x_o = p_articulacion[0]
articulacion_y_o = p_articulacion[1]
articulacion_z_o = p_articulacion[2]

style.use('ggplot')
ax1 = fig.add_subplot(111, projection='3d')

#Graficamos los brazos
puntos_funcion = np.transpose(puntos_funcion) 
for i in range(len(sec_coord_articulares)):
	
	#Nueva posicion para graficar
	x_IK = sec_coord_articulares[i][0]
	y_IK = sec_coord_articulares[i][1]
	z_IK = sec_coord_articulares[i][2]
	
	#Establecemos los limites del grafico
	ax1.set_xlim([-(a[1]+a[2]),a[1]+a[2]])
	ax1.set_ylim([-(a[1]+a[2]),a[1]+a[2]])
	
	# Graficamos la posicion inicial
	ax1.plot(articulacion_x_o,articulacion_y_o,articulacion_z_o,color='blue')
	start_joints=ax1.scatter(articulacion_x_o,articulacion_y_o,articulacion_z_o,label='start',color='blue')

	#Graficamos los puntos de la trayectoria
	puntos_trayectoria = ax1.scatter(puntos_funcion[0],puntos_funcion[1],puntos_funcion[2],color='green')
	
	#Graficamos la 
	nueva_posicion=ax1.scatter(x_IK,y_IK,z_IK,color='red')
	ax1.plot(x_IK,y_IK,z_IK,label='IK position',color='red')
	
	#Colcamos las leyendas
	plt.legend([start_joints,nueva_posicion, puntos_trayectoria], ['Posicion inicial del robot','Posicion en trayectoria', 'Puntos de la trayectoria'])
	plt.xlabel("x")
	plt.ylabel("y")
	plt.pause(0.001)
	#Para que no se borre el tapiz al llegar la ultima posicion
	if i != len(sec_coord_articulares)-1:
		ax1.clear()

plt.show()








