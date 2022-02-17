import os

import pandas as pd
import seaborn as sns
import streamlit as st

st.set_option("deprecation.showPyplotGlobalUse", False)


def file_selector(folder_path: str = "./datasets") -> str:
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select a file", filenames)
    return os.path.join(folder_path, selected_filename)


def show_dataset(df: pd.DataFrame) -> None:
    if st.checkbox("Show DataSet"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select", all_columns)
        number = st.number_input("Number of Rows to View", step=1)
        st.dataframe(df[selected_columns].head(int(number)))


def show_column_names(df: pd.DataFrame) -> None:
    if st.button("Columns Names"):
        st.write(df.columns.tolist())


def show_shape(df: pd.DataFrame) -> None:
    if st.button("Shape of Dataset"):
        st.write(df.shape)


def show_types(df: pd.DataFrame) -> None:
    if st.button("Data Types"):
        st.write(df.dtypes.to_dict())


def show_value_counts(df: pd.DataFrame) -> None:
    if st.button("Value Counts"):
        st.text("Value Counts By Column")
        st.write(df.iloc[:, -1].value_counts())


def show_summary(df: pd.DataFrame) -> None:
    if st.button("Summary"):
        st.write(df.describe())


def show_data(df: pd.DataFrame) -> None:
    show_shape(df)
    show_column_names(df)
    show_value_counts(df)
    show_summary(df)
    show_dataset(df)


def plot_corr(df):
    if st.checkbox("Correlation Chart"):
        st.write(sns.heatmap(df.corr(), annot=True))
        st.pyplot()


def plot_barplot(df: pd.DataFrame) -> None:
    if st.checkbox("Plot of Value Counts"):
        st.text("Value Counts By Target/Class")

        all_columns_names = df.columns.tolist()
        primary_col = st.selectbox("Select Primary Column To Group By", all_columns_names)
        selected_column_names = st.multiselect("Select Columns", all_columns_names)
        fig_size = st.slider("Select a scale (%)", 0.0, 200.0, 100.0) / 100

        if st.button("Plot"):
            st.text("Generating Plot for: {} and {}".format(primary_col, selected_column_names))

            if selected_column_names:
                vc_plot = df.groupby(primary_col)[selected_column_names].count()
            else:
                vc_plot = df.iloc[:, -1].value_counts()

            st.write(vc_plot.plot(kind="bar", figsize=(10 * fig_size, 10 * fig_size)))
            st.pyplot()


def plot_pieplot(df: pd.DataFrame) -> None:
    if st.checkbox("Pie Chart"):
        all_columns_names = df.columns.tolist()
        int_column = st.selectbox("Select Int Columns For Pie Plot", all_columns_names)
        selected_column_names = st.multiselect("Select Columns to sum up values", all_columns_names)
        fig_size = st.slider("Select a scale (%)", 0.0, 200.0, 100.0) / 100

        if st.button("Generate Pie Plot"):
            cust_values = df.groupby(selected_column_names)[int_column].agg('sum')
            st.write(cust_values.plot.pie(autopct="%1.1f%%", figsize=(10 * fig_size, 10 * fig_size)))
            st.pyplot()


def plot_barh(df: pd.DataFrame) -> None:
    if st.checkbox("BarH Chart"):
        all_columns_names = df.columns.tolist()
        x_column = st.selectbox("Select X Columns For Barh Plot", all_columns_names)
        y_column = st.selectbox("Select Y Columns For Barh Plot", all_columns_names)
        fig_size = st.slider("Select a scale (%)", 0.0, 200.0, 100.0) / 100

        barh_plot = df.plot.barh(x=x_column, y=y_column, figsize=(10 * fig_size, 10 * fig_size))
        if st.button("Generate Barh Plot"):
            st.write(barh_plot)
            st.pyplot()


def plot_custom(df: pd.DataFrame) -> None:
    if st.checkbox("Custom Chart"):
        all_columns_names = df.columns.tolist()
        type_of_plot = st.selectbox("Select the Type of Plot", ["area", "bar", "line", "hist", "box", "kde"])
        selected_column_names = st.multiselect("Select Columns To Plot", all_columns_names)

        if st.button("Generate Plot"):
            st.success("Generating A Customizable Plot of: {} for :: {}".format(type_of_plot, selected_column_names))

            if type_of_plot == "area":
                cust_data = df[selected_column_names]
                st.area_chart(cust_data)
            elif type_of_plot == "bar":
                cust_data = df[selected_column_names]
                st.bar_chart(cust_data)
            elif type_of_plot == "line":
                cust_data = df[selected_column_names]
                st.line_chart(cust_data)
            elif type_of_plot == "hist":
                custom_plot = df[selected_column_names].plot(kind=type_of_plot, bins=2)
                st.write(custom_plot)
                st.pyplot()
            elif type_of_plot == "box":
                custom_plot = df[selected_column_names].plot(kind=type_of_plot)
                st.write(custom_plot)
                st.pyplot()
            elif type_of_plot == "kde":
                custom_plot = df[selected_column_names].plot(kind=type_of_plot)
                st.write(custom_plot)
                st.pyplot()
            else:
                cust_plot = df[selected_column_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()


def show_plots(df: pd.DataFrame) -> None:
    plot_corr(df)
    plot_barplot(df)
    plot_pieplot(df)
    plot_barh(df)
    plot_custom(df)


def discover_features(df: pd.DataFrame) -> None:
    if st.checkbox("Show Features"):
        all_features = df.iloc[:, 0:-1]
        st.text("Features Names:: {}".format(all_features.columns[0:-1]))
        st.dataframe(all_features.head(10))

    if st.checkbox("Show Target"):
        all_target = df.iloc[:, -1]
        st.text("Target/Class Name:: {}".format(all_target.name))
        st.dataframe(all_target.head(10))


def main():
    st.title("Dataset Explorer")
    st.info("Data Explorer makes datasets easy to explore, visualize and communicate")
    filename = file_selector()

    st.write(f"You selected `{filename}`")
    df = pd.read_csv(filename)

    st.subheader("Exploratory Data Analysis")
    st.info("EDA is used to get acquainted with the data. More info: "
            "https://luminousmen.com/post/exploratory-data-analysis")
    show_data(df)

    st.subheader("Data Visualization")
    st.info("Plot charts")
    show_plots(df)

    st.subheader("Features/Target discovery")
    st.info("Explore ML related features")
    discover_features(df)


if __name__ == "__main__":
    main()
