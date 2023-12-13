import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import scipy.signal as signal
import scipy.stats as stats
import math
import time

fig1, (grafica1) = plt.subplots(1)
fig2, (grafica2) = plt.subplots(1)
fig3, (grafica3,grafica4) = plt.subplots(2,1)

st.title('Modelado de Sistemas')

columnas_graficas = st.columns(2)

st.sidebar.subheader('Seleccione un dominio del tiempo')
useropt = st.sidebar.selectbox('',['','Continuo','Discreto'])

r = 0.001
if useropt == 'Continuo':
    st.sidebar.subheader('Seleccione una señal de entrada')
    useroption = st.sidebar.selectbox('',['','Cuadrática','Sinusoidal','Triangular','Logarítmica base 10','Rampa','A trozos'])

    if useroption == 'Cuadrática':
        st.sidebar.latex('x(t)= at^2+bt+c')
        valor_a = st.sidebar.number_input('Ingrese el valor de a: ',None,None)
        valor_b = st.sidebar.number_input('Ingrese el valor de b: ',None,None)
        valor_c = st.sidebar.number_input('Ingrese el valor de c: ',None,None)
        a = st.sidebar.number_input('Ingrese el tiempo minimo en el eje x: ', None, None)
        b = st.sidebar.number_input('Ingrese el tiempo maximo en el eje x: ', None, None)
        #tminy = st.sidebar.number_input('Ingrese el valor minimo en el eje y donde desea observar la señal: ', None, None)
        #tmaxy = st.sidebar.number_input('Ingrese el valor maximo en el eje y donde desea observar la señal: ', None, None)
        graficar = st.sidebar.button('Graficar')

        try:
            valor_a = float(valor_a)
            valor_b = float(valor_b)
            valor_c = float(valor_c)
            xtot = np.arange(a,b+r,r)
            senal = (valor_a*(xtot**2))+(valor_b*xtot)+valor_c

            if graficar:
                grafica1.plot(xtot,senal,'g')
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                #grafica1.title('Gráfica función cuadrática')
                #plt.ylim(tminy,tmaxy)
                #grafica1.xlabel('tiempo')
                #plt.ylabel('x(t)')
                #plt.show()
                #stg = st.pyplot(plt)

        except:
            st.sidebar.write('Los valores de la función deben ser números')

    if useroption == 'Sinusoidal':
        st.sidebar.latex('x(t)= A*sin(2*\pi*f*t)')
        #st.sidebar.write('Ingresar frecuencia y amplitud' )
        A = st.sidebar.number_input('Ingrese el valor de la amplitud', key = "<uniquevalueofsomesort1>")
        f = st.sidebar.number_input('Ingrese el valor de la frecuencia', key = "<uniquevalueofsomesort2>")
        a = st.sidebar.number_input('Ingrese tiempo de inicio')
        graficar = st.sidebar.button('Graficar', key = "<uniquevalueofsomesort878872>")

        try:

            A = float(A)
            f = float(f)
            a = float(a)
            b = a+(1/f)
            #T = 1/f
            #r = 1/(4*2*np.pi*f)
            #r = r/f
            xtot = np.arange(a,b+r,r)
            senal = A*(np.sin(2*np.pi*f*xtot))

            if graficar:
                #grafica1.subplot(2,1,1)
                grafica1.plot(xtot,senal,'g')
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                #plt.xlim(a,T)
                #plt.title('Grafica función seno')
                #plt.xlabel('Tiempo')
                #plt.ylabel('sin(t)')
                #plt.show()
            
                #grafica1.show()
                
                #st_g.pyplot(plt)
        except:
            st.sidebar.write('La amplitud y la frecuencia deben ser números')

    if useroption == 'Triangular':
        st.sidebar.latex('x(t)= sawtooth(2*\pi*f*t)')
        fs = st.sidebar.number_input('Ingrese la frecuencia',None,None)
        A = st.sidebar.number_input('Ingrese la amplitud',None,None,0)
        a = st.sidebar.number_input('Ingrese el tiempo de inicio',None,None)
        #b = st.sidebar.number_input('Ingrese el tiempo final',None,None)
        graficar = st.sidebar.button('Graficar')

        try:
            fs = float(fs)
            A = int(A)
            a = float(a)
            b = a+(1/fs)
            #b = float(b)

            #n = 2000
            xtot = np.arange(a, b+r, r)
            senal = A*signal.sawtooth(2*np.pi*fs*xtot, 0.5)

            if graficar:
                grafica1.plot(xtot,senal,'g')
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                #plt.xlim(a, b)
                #plt.ylim(-A,A)
                #plt.title('Señal Triangular')
                #plt.xlabel('Tiempo')
                #plt.ylabel('Amplitud')
                #fig1.show()
                
        except:
            st.sidebar.write('Error')

    if useroption == 'Logarítmica base 10':
        st.sidebar.latex('x(t)= A*log10(t)')
        A = st.sidebar.number_input('Ingrese el valor de la amplitud: ', None,None)
        a = st.sidebar.number_input('Ingrese el tiempo minimo donde desea ver la gráfica: ', None,None)
        b = st.sidebar.number_input('Ingrese el tiempo maximo donde desea ver la gráfica: ', None, None)
        graficar = st.sidebar.button('Graficar')

        try:
            A = float(A)
            xtot = np.arange(a,b+r,r)
            senal = A*np.log10(xtot)

            if graficar:
                grafica1.plot(xtot,senal,'g')
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                #plt.title('Gráfica función logaritmica en base 10')
                #plt.xlabel('Tiempo')
                #plt.ylabel('Amplitud')
                #plt.show()
        except:
            st.sidebar.write('Error')

    if useroption == 'Rampa':
        st.sidebar.latex('x(t)=')
        st.sidebar.latex('t-a, a \le t < t1')
        st.sidebar.latex('t, t1 \le t \le t2')
        st.sidebar.latex('t+a, t2 < t \le b')
        a = st.sidebar.number_input('Ingrese el tiempo de inicio', None,None)
        b = st.sidebar.number_input('Ingrese el tiempo final',None,None)
        graficar = st.sidebar.button('Graficar')

        try:
            rest = b-a
            div = rest/3
            pend = 1
            xtot = np.arange(a,b+r,r)

            x1 = np.arange(a,a+div+r,r)
            y1 = (pend*x1)-a

            x3 = np.arange(b-div,b+r,r)
            y3 = (-pend*x3)+b

            G = len(xtot)
            senal = np.zeros(G)
            senal[xtot<=a+div+r] = y1
            senal[xtot>=a+div+r] = pend*div
            senal[xtot>=a+2*div-r] = y3
            #senal = np.resize(senal,np.shape(xtot))
            if graficar:
                grafica1.plot(xtot,senal,'g')
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
        except:
            st.sidebar.write('Error')


    if useroption == 'A trozos':
        st.sidebar.latex('x(t)=')
        st.sidebar.latex('-20*t+t1, a \le t < t1')
        st.sidebar.latex('30, t1 \le t < t2')
        st.sidebar.latex('e^t, t2 \le t \le b')
        st.sidebar.write('Ingresar intervalos' )
        a = st.sidebar.number_input('Ingrese el tiempo de inicio', key = "<uniquevalueofsomesortfrrer>")
        t1 = st.sidebar.number_input('Ingrese el tiempo final de la primera función')
        t2 = st.sidebar.number_input('Ingrese el tiempo final de la segunda función')
        b = st.sidebar.number_input('Ingrese el tiempo final')
        graficar = st.sidebar.button('Graficar')

        try:
            xtot = np.arange(a,b+r,r)

            x1 = np.arange(a,t1+r,r)
            y1 = -20*x1+t1
            #grafica1.plot(x1,y1)

            #desp = c-b
            #x2 = np.arange(t1,t2+r,r)
            #y2 = -20*x2+1
            #y2 = np.log10(x2)
            #grafica1.plot(x2,y2)

            x3 = np.arange(t2,b+r,r)
            y3 = np.exp(x3)
            #grafica1.plot(x3,y3)

            G = len(xtot)
            senal = np.zeros(G)
            senal[xtot<t1+r] = y1
            senal[xtot>t1] = 30
            #senal[xtot<=t1+r] = 30
            senal[xtot>=t2] = y3
            

            if graficar:
                #grafica1.plot(x1,y1)
                #grafica1.plot(x2,y2)
                #grafica1.plot(x3,y3)
                grafica1.plot(xtot,senal,'g')
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                #plt.title('Grafica función a trozos')
                #plt.xlabel('Tiempo')
                #plt.ylabel('Función a trozos')
                #plt.show()
                #st_g.pyplot(plt)
                
        except:
            st.sidebar.write('Error')

    




    #IMPULSO EMPIEZA AQUI




    st.sidebar.subheader('Seleccione una señal impulso')
    useroption2 = st.sidebar.selectbox(' ',['','Cuadrática','Sinusoidal','Triangular','Logarítmica base 10','Rampa','A trozos'])

    if useroption2 == 'Cuadrática':
        st.sidebar.latex('h(t)= at^2+bt+c')
        valor_a2 = st.sidebar.number_input('Ingrese el valor de a',None,None)
        valor_b2 = st.sidebar.number_input('Ingrese el valor de b',None,None)
        valor_c2 = st.sidebar.number_input('Ingrese el valor de c',None,None)
        c = st.sidebar.number_input('Ingrese el tiempo minimo en el eje x', None, None)
        d = st.sidebar.number_input('Ingrese el tiempo maximo en el eje x', None, None)
        #tminy2 = st.sidebar.number_input('Ingrese el valor minimo en el eje y donde desea observar la señal', None, None)
        #tmaxy2 = st.sidebar.number_input('Ingrese el valor maximo en el eje y donde desea observar la señal', None, None)
        graficar2 = st.sidebar.button('Ok')

        try:
            valor_a2 = float(valor_a2)
            valor_b2 = float(valor_b2)
            valor_c2 = float(valor_c2)
            xtot2 = np.arange(c,d+r,r)
            y_h = (valor_a2*(xtot2**2))+(valor_b2*xtot2)+valor_c2

            if graficar2:
                grafica1.plot(xtot,senal,'g')
                grafica2.plot(xtot2,y_h,'r')
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                grafica2.axis(xmin=c,xmax=d)
                grafica2.grid(None,'both','both')
                #plt.title('Gráfica función cuadrática')
                #plt.ylim(tminy,tmaxy)
                #plt.xlabel('tiempo')
                #plt.ylabel('x(t)')
                #plt.show()
                #stg = st.pyplot(plt)

        except:
            st.sidebar.write('Los valores de la función deben ser números')


    if useroption2 == 'Sinusoidal':
        st.sidebar.latex('h(t)= A*sin(2*\pi*f*t)')
        #st.sidebar.write('Ingresar frecuencia y amplitud' )
        A2 = st.sidebar.number_input('Ingrese el valor de la amplitud', key = "<uniquevalueofsomesort3>")
        f2 = st.sidebar.number_input('Ingrese el valor de la frecuencia', key = "<uniquevalueofsomesort4>")
        c = st.sidebar.number_input('Ingrese tiempo de inicio ')
        graficar2 = st.sidebar.button('Ok')

        try:
            A2 = float(A2)
            f2 = float(f2)
            c = float(c)
            d = c+(1/f2)
            #T2 = 1/f2
            #r2 = 1/(4*2*np.pi*f2)
            #r = r/f
            xtot2 = np.arange(c,d+r,r)
            #xtot = np.arange(a,b+r,r)
            y_h = A2*(np.sin(2*np.pi*f2*xtot2))
            #senal = A*(np.sin(2*np.pi*f*xtot))
            if graficar2:
                #grafica1.subplot(2,1,1)
                grafica1.plot(xtot,senal,'g')
                grafica2.plot(xtot2,y_h,'r')
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                grafica2.axis(xmin=c,xmax=d)
                grafica2.grid(None,'both','both')
                #plt.title('Grafica función seno')
                #plt.xlabel('Tiempo')
                #plt.ylabel('sin(t)')
                #plt.show()
                
                #grafica1.show()
            
                #st_g.pyplot(plt)
        except:
            st.sidebar.write('La amplitud y la frecuencia deben ser números')

    if useroption2 == 'Triangular':
        st.sidebar.latex('h(t)= sawtooth(2*\pi*f*t)')
        fs2 = st.sidebar.number_input('Ingrese la frecuencia ',None,None)
        A2 = st.sidebar.number_input('Ingrese la amplitud ',None,None,0)
        c = st.sidebar.number_input('Ingrese el tiempo de inicio ',None,None)
        d = st.sidebar.number_input('Ingrese el tiempo final ',None,None)
        graficar2 = st.sidebar.button('Ok')

        try:
            fs2=float(fs2)
            #A2=int(A2)
            c=float(c)
            d=float(d)
            n2 = 2000
            xtot2 = np.arange(c,d+r,r)
            y_h = A2*signal.sawtooth(2*np.pi*fs2*xtot2, 0.5)

            if graficar2:
                grafica1.plot(xtot,senal,'g')
                grafica2.plot(xtot2,y_h,'r')
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                grafica2.axis(xmin=c,xmax=d)
                grafica2.grid(None,'both','both')
                #plt.xlim(c, d)
                #plt.ylim(-A2,A2)
                #plt.title('Señal Triangular')
                #plt.xlabel('Tiempo')
                #plt.ylabel('Amplitud')
                #fig2.show()
                
        except:
            st.sidebar.write('Error')

    if useroption2 == 'Logarítmica base 10':
        st.sidebar.latex('h(t)= A*log10(t)')
        A2 = st.sidebar.number_input('Ingrese el valor de la amplitud', None,None)
        c = st.sidebar.number_input('Ingrese el tiempo minimo donde desea ver la gráfica', None,None)
        d = st.sidebar.number_input('Ingrese el tiempo maximo donde desea ver la gráfica', None, None)
        graficar2 = st.sidebar.button('Ok')

        try:
            A2 = float(A2)
            xtot2 = np.arange(c,d+r,r)
            y_h = A2*np.log10(xtot2)

            if graficar2:
                grafica1.plot(xtot,senal,'g')
                grafica2.plot(xtot2,y_h,'r')
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                grafica2.axis(xmin=c,xmax=d)
                grafica2.grid(None,'both','both')
                #plt.title('Gráfica función logaritmica en base 10')
                #plt.xlabel('Tiempo')
                #plt.ylabel('Amplitud')
                #plt.show()
        except:
            st.sidebar.write('Error 2')

    if useroption2 == 'Rampa':
        st.sidebar.latex('h(t)=')
        st.sidebar.latex('t-a, a \le t < t1')
        st.sidebar.latex('t, t1 \le t \le t2')
        st.sidebar.latex('t+a, t2 < t \le b')
        c = st.sidebar.number_input('Ingrese el tiempo de inicio ', None,None)
        d = st.sidebar.number_input('Ingrese el tiempo final ',None,None)
        graficar2 = st.sidebar.button('Ok')

        try:
            rest2 = d-c
            div2 = rest2/3
            pend2 = 1
            xtot2 = np.arange(c,d+r,r)

            x1_2 = np.arange(c,c+div2+r,r)
            y1_2 = (pend2*x1_2)-c

            x3_2 = np.arange(d-div2,d+r,r)
            y3_2 = (-pend2*x3_2)+d

            G2 = len(xtot2)
            y_h = np.zeros(G2)
            y_h[xtot2<=c+div2+r] = y1_2
            y_h[xtot2>=c+div2+r] = pend2*div2
            y_h[xtot2>=c+2*div2-r] = y3_2

            if graficar2:
                grafica1.plot(xtot,senal,'g')
                grafica2.plot(xtot2,y_h,'r')
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                grafica2.axis(xmin=c,xmax=d)
                grafica2.grid(None,'both','both')

        except:
            st.sidebar.write('Error')

    if useroption2 == 'A trozos':
        st.sidebar.latex('h(t)= ')
        st.sidebar.latex('-20*t+t1, a \le t < t1')
        st.sidebar.latex('30, t1 \le t < t2')
        st.sidebar.latex('e^t, t2 \le t \le b')
        st.sidebar.write('Ingresar intervalos' )
        c = st.sidebar.number_input('Ingrese el tiempo de inicio ')
        b2 = st.sidebar.number_input('Ingrese el tiempo final de la primera función ')
        t1_2 = st.sidebar.number_input('Ingrese el tiempo final de la segunda función ')
        d = st.sidebar.number_input('Ingrese el tiempo final ')
        graficar2 = st.sidebar.button('Ok')

        try:
            xtot2 = np.arange(c,d+r,r)

            x1_2 = np.arange(c,b2+r,r)
            y1_2 = -20*x1_2+b2
            #grafica1.plot(x1,y1)

            #desp = c-b
            #x2 = np.arange(b,c,r)
            #y2 = -20*x2+1
            #y2 = np.log10(x2)
            #grafica1.plot(x2,y2)

            x3_2 = np.arange(t1_2,d+r,r)
            y3_2 = np.exp(x3_2)
            #grafica1.plot(x3,y3)

            G2 = len(xtot2)
            y_h = np.zeros(G2)
            y_h[xtot2<=b2+r] = y1_2
            y_h[xtot2>b2] = 30
            y_h[xtot2>=t1_2] = y3_2


            if graficar2:
                #grafica1.plot(x1,y1)
                #grafica1.plot(x2,y2)
                #grafica1.plot(x3,y3)
                grafica1.plot(xtot,senal,'g')
                grafica2.plot(xtot2,y_h,'r')
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                grafica2.axis(xmin=c,xmax=d)
                grafica2.grid(None,'both','both')
                #plt.title('Grafica función a trozos')
                #plt.xlabel('Tiempo')
                #plt.ylabel('Función a trozos')
                #plt.show()
                #st_g.pyplot(plt)
                
        except:
            st.sidebar.write('Error')




#TIEMPO DISCRETO



#r_dis = 0.05 
if useropt == 'Discreto':
    st.sidebar.subheader('Seleccione una señal de entrada')
    useroption = st.sidebar.selectbox('',['','Cuadrática','Sinusoidal','Triangular','Logarítmica base 10','Rampa','A trozos'])

    if useroption == 'Cuadrática':
        st.sidebar.latex('x[n]= an^2+bn+c')
        valor_a = st.sidebar.number_input('Ingrese el valor de a: ',None,None)
        valor_b = st.sidebar.number_input('Ingrese el valor de b: ',None,None)
        valor_c = st.sidebar.number_input('Ingrese el valor de c: ',None,None)
        a = st.sidebar.number_input('Ingrese el tiempo minimo en el eje x: ', None, None)
        b = st.sidebar.number_input('Ingrese el tiempo maximo en el eje x: ', None, None)
        #tminy = st.sidebar.number_input('Ingrese el valor minimo en el eje y donde desea observar la señal: ', None, None)
        #tmaxy = st.sidebar.number_input('Ingrese el valor maximo en el eje y donde desea observar la señal: ', None, None)
        graficar = st.sidebar.button('Graficar')

        try:
            valor_a = float(valor_a)
            valor_b = float(valor_b)
            valor_c = float(valor_c)
            r_dis = (b-a)/30
            xtot = np.arange(a,b+r_dis,r_dis)
            senal = (valor_a*(xtot**2))+(valor_b*xtot)+valor_c

            if graficar:
                grafica1.stem(xtot,senal)
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                #plt.title('Gráfica función cuadrática')
                #plt.ylim(tminy,tmaxy)
                #plt.xlabel('tiempo')
                #plt.ylabel('x(t)')
                #plt.show()
                #stg = st.pyplot(plt)

        except:
            st.sidebar.write('Los valores de la función deben ser números')

    if useroption == 'Sinusoidal':
        st.sidebar.latex('x[n]= A*sin(2*\pi*f*n)')
        #st.sidebar.write('Ingresar frecuencia y amplitud' )
        A = st.sidebar.number_input('Ingrese el valor de la amplitud', key = "<uniquevalueofsomesort1>")
        f = st.sidebar.number_input('Ingrese el valor de la frecuencia', key = "<uniquevalueofsomesort2>")
        a = st.sidebar.number_input('Ingrese tiempo de inicio')
        graficar = st.sidebar.button('Graficar', key = "<uniquevalueofsomesort878872>")

        try:

            A = float(A)
            f = float(f)
            a = float(a)
            b = a+(1/f)
            T = 1/f
            #r = 1/(4*2*np.pi*f)
            #r = r/f
            r_dis = (b-a)/30
            xtot = np.arange(a,b+r_dis,r_dis)
            senal = A*(np.sin(2*np.pi*f*xtot))

            if graficar:
                #grafica1.subplot(2,1,1)
                grafica1.stem(xtot,senal)
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                #plt.xlim(a,T)
                #plt.title('Grafica función seno')
                #plt.xlabel('Tiempo')
                #plt.ylabel('sin(t)')
                #plt.show()
            
                #grafica1.show()
                
                #st_g.pyplot(plt)
        except:
            st.sidebar.write('La amplitud y la frecuencia deben ser números')

    if useroption == 'Triangular':
        st.sidebar.latex('x[n]= sawtooth(2*\pi*f*n)')
        fs = st.sidebar.number_input('Ingrese la frecuencia',None,None)
        A = st.sidebar.number_input('Ingrese la amplitud',None,None,0)
        a = st.sidebar.number_input('Ingrese el tiempo de inicio',None,None)
        #b = st.sidebar.number_input('Ingrese el tiempo final',None,None)
        graficar = st.sidebar.button('Graficar')

        try:
            fs = float(fs)
            A = int(A)
            a = float(a)
            b= a+(1/fs)
            #b = float(b)
            #n = 2000
            r_dis = (b-a)/30
            xtot = np.arange(a, b+r_dis, r_dis)
            senal = A*signal.sawtooth(2*np.pi*fs*xtot, 0.5)

            if graficar:
                grafica1.stem(xtot,senal)
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                #plt.xlim(a, b)
                #plt.ylim(-A,A)
                #plt.title('Señal Triangular')
                #plt.xlabel('Tiempo')
                #plt.ylabel('Amplitud')
                #fig1.show()
                
        except:
            st.sidebar.write('Error')

    if useroption == 'Logarítmica base 10':
        st.sidebar.latex('x[n]= A*log10(n)')
        A = st.sidebar.number_input('Ingrese el valor de la amplitud: ', None,None)
        a = st.sidebar.number_input('Ingrese el tiempo minimo donde desea ver la gráfica: ', None,None)
        b = st.sidebar.number_input('Ingrese el tiempo maximo donde desea ver la gráfica: ', None, None)
        graficar = st.sidebar.button('Graficar')

        try:
            A = float(A)
            r_dis = (b-a)/30
            xtot = np.arange(a,b+r_dis,r_dis)
            senal = A*np.log10(xtot)

            if graficar:
                grafica1.stem(xtot,senal)
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                #plt.title('Gráfica función logaritmica en base 10')
                #plt.xlabel('Tiempo')
                #plt.ylabel('Amplitud')
                #plt.show()
        except:
            st.sidebar.write(' ')

    if useroption == 'Rampa':
        st.sidebar.latex('x[n]=')
        st.sidebar.latex('n-a, a \le n < n1')
        st.sidebar.latex('n, n1 \le n \le n2')
        st.sidebar.latex('n+a, n2 < n \le b')
        a = st.sidebar.number_input('Ingrese el tiempo de inicio', None,None)
        b = st.sidebar.number_input('Ingrese el tiempo final',None,None)
        graficar = st.sidebar.button('Graficar')

        try:
            rest = b-a
            div = rest/3
            pend = 1
            r_dis = (b-a)/30
            xtot = np.arange(a,b+r_dis,r_dis)

            x1 = np.arange(a,a+div+r_dis,r_dis)
            y1 = (pend*x1)-a

            x3 = np.arange(b-div,b+r_dis,r_dis)
            y3 = (-pend*x3)+b

            G = len(xtot)
            senal = np.zeros(G)
            senal[xtot<=a+div] = y1
            senal[xtot>=a+div+r_dis] = pend*div
            senal[xtot>=a+2*div] = y3

            if graficar:
                grafica1.stem(xtot,senal)
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
        except:
            st.sidebar.write('Error')


    if useroption == 'A trozos':
        st.sidebar.latex('x[n]=')
        st.sidebar.latex('-20*n+n1, a \le n < n1')
        st.sidebar.latex('30, n1 \le n < n2')
        st.sidebar.latex('e^n, n2 \le n \le b')
        st.sidebar.write('Ingresar intervalos' )
        a = st.sidebar.number_input('Ingrese el tiempo de inicio', key = "<uniquevalueofsomesortfrrer>")
        t1 = st.sidebar.number_input('Ingrese el tiempo final de la primera función')
        t2 = st.sidebar.number_input('Ingrese el tiempo final de la segunda función')
        b = st.sidebar.number_input('Ingrese el tiempo final')
        graficar = st.sidebar.button('Graficar')

        try:
            r_dis = (b-a)/30
            xtot = np.arange(a,b+r_dis,r_dis)

            x1 = np.arange(a,t1+r_dis,r_dis)
            y1 = -20*x1+t1
            #grafica1.plot(x1,y1)

            #desp = c-b
            #x2 = np.arange(t1,t2+r,r)
            #y2 = -20*x2+1
            #y2 = np.log10(x2)
            #grafica1.plot(x2,y2)

            x3 = np.arange(t2,b+r_dis,r_dis)
            y3 = np.exp(x3)
            #grafica1.plot(x3,y3)

            G = len(xtot)
            senal = np.zeros(G)
            senal[xtot<=t1] = y1
            senal[xtot>t1] = 30
            #senal[xtot<=t1+r] = 30
            senal[xtot>=t2] = y3


            if graficar:
                #grafica1.plot(x1,y1)
                #grafica1.plot(x2,y2)
                #grafica1.plot(x3,y3)
                grafica1.stem(xtot,senal)
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                #plt.title('Grafica función a trozos')
                #plt.xlabel('Tiempo')
                #plt.ylabel('Función a trozos')
                #plt.show()
                #st_g.pyplot(plt)
                
        except:
            st.sidebar.write('Error')

    




    #IMPULSO EMPIEZA AQUI




    st.sidebar.subheader('Seleccione una señal impulso')
    useroption2 = st.sidebar.selectbox(' ',['','Cuadrática','Sinusoidal','Triangular','Logarítmica base 10','Rampa','A trozos'])

    if useroption2 == 'Cuadrática':
        st.sidebar.latex('h[n]= an^2+bn+c')
        valor_a2 = st.sidebar.number_input('Ingrese el valor de a',None,None)
        valor_b2 = st.sidebar.number_input('Ingrese el valor de b',None,None)
        valor_c2 = st.sidebar.number_input('Ingrese el valor de c',None,None)
        c = st.sidebar.number_input('Ingrese el tiempo minimo en el eje x', None, None)
        d = st.sidebar.number_input('Ingrese el tiempo maximo en el eje x', None, None)
        #tminy2 = st.sidebar.number_input('Ingrese el valor minimo en el eje y donde desea observar la señal', None, None)
        #tmaxy2 = st.sidebar.number_input('Ingrese el valor maximo en el eje y donde desea observar la señal', None, None)
        graficar2 = st.sidebar.button('Ok')

        try:
            valor_a2 = float(valor_a2)
            valor_b2 = float(valor_b2)
            valor_c2 = float(valor_c2)
            r_dis = (d-c)/30
            xtot2 = np.arange(c,d+r_dis,r_dis)
            y_h = (valor_a2*(xtot2**2))+(valor_b2*xtot2)+valor_c2

            if graficar2:
                grafica1.stem(xtot,senal)
                grafica2.stem(xtot2,y_h)
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                grafica2.axis(xmin=c,xmax=d)
                grafica2.grid(None,'both','both')
                #plt.title('Gráfica función cuadrática')
                #plt.ylim(tminy,tmaxy)
                #plt.xlabel('tiempo')
                #plt.ylabel('x(t)')
                #plt.show()
                #stg = st.pyplot(plt)

        except:
            st.sidebar.write('Los valores de la función deben ser números')


    if useroption2 == 'Sinusoidal':
        st.sidebar.latex('h[n]= A*sin(2*\pi*f*n)')
        #st.sidebar.write('Ingresar frecuencia y amplitud' )
        A2 = st.sidebar.number_input('Ingrese el valor de la amplitud', key = "<uniquevalueofsomesort3>")
        f2 = st.sidebar.number_input('Ingrese el valor de la frecuencia', key = "<uniquevalueofsomesort4>")
        c = st.sidebar.number_input('Ingrese tiempo de inicio ')
        graficar2 = st.sidebar.button('Ok')

        try:
            A2 = float(A2)
            f2 = float(f2)
            c = float(c)
            d = c+(1/f2)
            T2 = 1/f2
            r_dis = (d-c)/30
            #r2 = 1/(4*2*np.pi*f2)
            #r = r/f
            xtot2 = np.arange(c,d+r_dis,r_dis)
            #xtot = np.arange(a,b+r,r)
            y_h = A2*(np.sin(2*np.pi*f2*xtot2))
            #senal = A*(np.sin(2*np.pi*f*xtot))
            if graficar2:
                #grafica1.subplot(2,1,1)
                grafica1.stem(xtot,senal)
                grafica2.stem(xtot2,y_h)
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                grafica2.axis(xmin=c,xmax=d)
                grafica2.grid(None,'both','both')
                #plt.title('Grafica función seno')
                #plt.xlabel('Tiempo')
                #plt.ylabel('sin(t)')
                #plt.show()
                
                #grafica1.show()
            
                #st_g.pyplot(plt)
        except:
            st.sidebar.write('La amplitud y la frecuencia deben ser números')

    if useroption2 == 'Triangular':
        st.sidebar.latex('h[n]= sawtooth2*\pi*f*n)')
        fs2 = st.sidebar.number_input('Ingrese la frecuencia ',None,None)
        A2 = st.sidebar.number_input('Ingrese la amplitud ',None,None,0)
        c = st.sidebar.number_input('Ingrese el tiempo de inicio ',None,None)
        #d = st.sidebar.number_input('Ingrese el tiempo final ',None,None)
        graficar2 = st.sidebar.button('Ok')

        try:
            fs2=float(fs2)
            #A2=int(A2)
            c=float(c)
            d= c+(1/fs2)
            r_dis = (d-c)/30
            #d=float(d)
            #n2 = 2000
            xtot2 = np.arange(c,d+r_dis,r_dis)
            y_h = A2*signal.sawtooth(2*np.pi*fs2*xtot2, 0.5)

            if graficar2:
                grafica1.stem(xtot,senal)
                grafica2.stem(xtot2,y_h)
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                grafica2.axis(xmin=c,xmax=d)
                grafica2.grid(None,'both','both')
                #plt.xlim(c, d)
                #plt.ylim(-A2,A2)
                #plt.title('Señal Triangular')
                #plt.xlabel('Tiempo')
                #plt.ylabel('Amplitud')
                #fig2.show()
                
        except:
            st.sidebar.write('Adios')

    if useroption2 == 'Logarítmica base 10':
        st.sidebar.latex('h[n]= A*log10(n)')
        A2 = st.sidebar.number_input('Ingrese el valor de la amplitud', None,None)
        c = st.sidebar.number_input('Ingrese el tiempo minimo donde desea ver la gráfica', None,None)
        d = st.sidebar.number_input('Ingrese el tiempo maximo donde desea ver la gráfica', None, None)
        graficar2 = st.sidebar.button('Ok')

        try:
            A2 = float(A2)
            r_dis = (d-c)/30
            xtot2 = np.arange(c,d+r_dis,r_dis)
            y_h = A2*np.log10(xtot2)

            if graficar2:
                grafica1.stem(xtot,senal)
                grafica2.stem(xtot2,y_h)
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                grafica2.axis(xmin=c,xmax=d)
                grafica2.grid(None,'both','both')
                #plt.title('Gráfica función logaritmica en base 10')
                #plt.xlabel('Tiempo')
                #plt.ylabel('Amplitud')
                #plt.show()
        except:
            st.sidebar.write('Error 2')

    if useroption2 == 'Rampa':
        st.sidebar.latex('h[n]=')
        st.sidebar.latex('n-a, a \le n < n1')
        st.sidebar.latex('n, n1 \le n \le n2')
        st.sidebar.latex('n+a, n2 < n \le b')
        c = st.sidebar.number_input('Ingrese el tiempo de inicio ', None,None)
        d = st.sidebar.number_input('Ingrese el tiempo final ',None,None)
        graficar2 = st.sidebar.button('Ok')

        try:
            rest2 = d-c
            div2 = rest2/3
            pend2 = 1
            r_dis = (d-c)/30
            xtot2 = np.arange(c,d+r_dis,r_dis)

            x1_2 = np.arange(c,c+div2+r_dis,r_dis)
            y1_2 = (pend2*x1_2)-c

            x3_2 = np.arange(d-div2,d+r_dis,r_dis)
            y3_2 = (-pend2*x3_2)+d

            G2 = len(xtot2)
            y_h = np.zeros(G2)
            y_h[xtot2<=c+div2] = y1_2
            y_h[xtot2>=c+div2+r_dis] = pend2*div2
            y_h[xtot2>=c+2*div2] = y3_2

            if graficar2:
                grafica1.stem(xtot,senal)
                grafica2.stem(xtot2,y_h)
                grafica1.axis(xmin=a,xmax=b)
                grafica1.grid(None,'both','both')
                grafica2.axis(xmin=c,xmax=d)
                grafica2.grid(None,'both','both')

        except:
            st.sidebar.write('Error')

    if useroption2 == 'A trozos':
        st.sidebar.latex('h[n]= ')
        st.sidebar.latex('-20*n+n1, a \le n < n1')
        st.sidebar.latex('30, n1 \le n < n2')
        st.sidebar.latex('e^n, n2 \le n \le b')
        st.sidebar.write('Ingresar intervalos' )
        c = st.sidebar.number_input('Ingrese el tiempo de inicio ')
        b2 = st.sidebar.number_input('Ingrese el tiempo final de la primera función ')
        t1_2 = st.sidebar.number_input('Ingrese el tiempo final de la segunda función ')
        d = st.sidebar.number_input('Ingrese el tiempo final ')
        graficar2 = st.sidebar.button('Ok')

        try:
            r_dis = (d-c)/30
            xtot2 = np.arange(c,d+r_dis,r_dis)

            x1_2 = np.arange(c,b2+r_dis,r_dis)
            y1_2 = -20*x1_2+b2
            #grafica1.plot(x1,y1)

            #desp = c-b
            #x2 = np.arange(b,c,r)
            #y2 = -20*x2+1
            #y2 = np.log10(x2)
            #grafica1.plot(x2,y2)

            x3_2 = np.arange(t1_2,d+r_dis,r_dis)
            y3_2 = np.exp(x3_2)
            #grafica1.plot(x3,y3)

            G2 = len(xtot2)
            y_h = np.zeros(G2)
            y_h[xtot2<=b2] = y1_2
            y_h[xtot2>b2] = 30
            y_h[xtot2>=t1_2] = y3_2


            if graficar2:
                #grafica1.plot(x1,y1)
                #grafica1.plot(x2,y2)
                #grafica1.plot(x3,y3)
                grafica1.stem(xtot,senal)
                grafica2.stem(xtot2,y_h)
                #plt.title('Grafica función a trozos')
                #plt.xlabel('Tiempo')
                #plt.ylabel('Función a trozos')
                #plt.show()
                #st_g.pyplot(plt)
                
        except:
            st.sidebar.write('Error')





#grafica1.plot(xtot,senal)
#grafica2.plot(xtot2,y_h)


with columnas_graficas[0]:
    st.subheader('Señal de entrada x(t) / x[n]')
    st_g = st.pyplot(fig1,False)

with columnas_graficas[1]:
    st.subheader('Señal impulso h(t) / h[n]')
    h_g = st.pyplot(fig2,False)



if useropt == 'Continuo':
    try:
        #st_g = st.pyplot(plt)
        grafica1.plot(xtot,senal,'g')
        grafica2.plot(xtot2,y_h,'r')
        grafica1.axis(xmin=a,xmax=b)
        grafica1.grid(None,'both','both')
        grafica2.axis(xmin=c,xmax=d)
        grafica2.grid(None,'both','both')
    except:
        st.write('')

if useropt == 'Discreto':
    try:
        #st_g = st.pyplot(plt)
        grafica1.stem(xtot,senal)
        grafica2.stem(xtot2,y_h)
        grafica1.axis(xmin=a,xmax=b)
        grafica1.grid(None,'both','both')
        grafica2.axis(xmin=c,xmax=d)
        grafica2.grid(None,'both','both')
    except:
        st.write('')

convol = st.button('Convolucionar')
conv_g = st.pyplot(fig3)

if useropt == 'Continuo':
    if convol:
        try:
            time1 = np.arange(a,b+r,r)
            time2 = np.arange(c,d+r,r)

            L = len(time1)
            M = len(time2)
            Y = M+L-1

            ty = np.arange(a+c,b+d+r,r)
            y_con = np.convolve(senal,y_h)*r
            st_g.pyplot(fig1)
            h_g.pyplot(fig2)
            y_con = np.resize(y_con,np.shape(ty))
            grafica3.plot(ty,y_con,'r')
        except:
            st.text('no sirve')
        
        hs = y_h[::-1]
        tm = np.arange(a-(d-c),a+r,r)
        hs = np.resize(hs,np.shape(tm))
        aniy = np.zeros(len(y_con))
        frames = 30
        factor = len(ty)/frames
        factor = math.floor(factor)
        factor_t = (max(ty)-min(ty))/frames

        for i in range(frames+1):
            aniy[:i*factor] = y_con[:i*factor]
            grafica3.clear()
            grafica3.plot(tm,hs,xtot,senal)
            grafica3.axis(xmin=min(ty), xmax=max(ty))
            grafica4.clear()
            grafica4.plot(ty,aniy)
            tm = tm + factor_t
            time.sleep(0.01)
            grafica3.legend(['h(t)','x(t)'])
            conv_g.pyplot(fig3)

if useropt == 'Discreto':
    if convol:
        try:
            time1 = np.arange(a,b+r_dis,r_dis)
            time2 = np.arange(c,d+r_dis,r_dis)

            L = len(time1)
            M = len(time2)
            Y = M+L-1

            ty = np.arange(a+c,b+d+r_dis,r_dis)
            #senal=np.resize(senal,np.shape(xtot))
            y_con = np.convolve(senal,y_h)*r_dis
            st_g.pyplot(fig1)
            h_g.pyplot(fig2)
            y_con = np.resize(y_con,np.shape(ty))
            grafica3.stem(ty,y_con,'r')
        except:
            st.text('no sirve')
        
        hs = y_h[::-1]
        tm = np.arange(a-(d-c),a+r_dis,r_dis)
        hs = np.resize(hs,np.shape(tm))
        #xtot=np.resize(xtot,np.shape(senal))
        aniy = np.zeros(len(y_con))
        frames = 30
        factor = len(ty)/frames
        factor = math.floor(factor)
        factor_t = (max(ty)-min(ty))/frames

        for i in range(frames+1):
            aniy[:i*factor] = y_con[:i*factor]
            grafica3.clear()
            grafica3.stem(tm,hs)
            grafica3.stem(xtot,senal)
            grafica3.axis(xmin=min(ty), xmax=max(ty))
            grafica4.clear()
            grafica4.stem(ty,aniy)
            tm = tm + factor_t
            time.sleep(0.01)
            grafica3.legend(['h(t)','x(t)'])
            conv_g.pyplot(fig3)
